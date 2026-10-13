"""Trinity-style pretraining optimization, with independent routed expert updates.

Muon uses PyTorch's original shape adjustment, lr * sqrt(max(1, out / in)).
Embeddings, output heads, and scalar/vector parameters use AdamW. FP32 master
weights and optimizer states preserve small updates while model compute is bf16.
"""

from dataclasses import dataclass

import torch

from src.model.lora import LoRAMLP
from src.model.moe import MoEMLP
from src.run.util.config import TrainConfig
from src.run.util.distributed import get_raw_model


@dataclass
class ParameterUpdate:
    """Map an FP32 master matrix/vector back to a model parameter or expert slice."""
    source: torch.nn.Parameter
    master: torch.nn.Parameter
    index: tuple | None
    expert: int | None


class PretrainingOptimizer:
    """Coordinate Muon/AdamW and copy their FP32 master updates to the model.

    Routed MLP slices are separate master matrices: both Muon's momentum and
    orthogonalization stay local to an expert. Absent experts skip momentum and
    weight decay as well as gradient updates. Source gradients must already be
    accumulated, globally reduced by DDP, and clipped before ``step``.
    """

    def __init__(self, model: torch.nn.Module, train: TrainConfig):
        self.model = model
        self.updates: list[ParameterUpdate] = []
        raw = get_raw_model(model)
        muon, adam_decay, adam_no_decay = [], [], []
        for name, param in raw.named_parameters():
            if not param.requires_grad:
                continue
            parent_name, leaf = name.rsplit(".", 1)
            parent = raw.get_submodule(parent_name)
            routed = hasattr(raw, "labels")
            owner = 0 if routed else None
            slices = [(None, owner)]

            # The MoE matrices concatenate experts along output rows (gate/up)
            # or input columns (down). Treat each slice as its own Muon matrix.
            if leaf == "weight" and parent_name.endswith((".fc_1", ".fc_2", ".proj")):
                mlp = raw.get_submodule(parent_name.rsplit(".", 1)[0])
                if isinstance(mlp, MoEMLP):
                    slices = [
                        ((slice(None), mlp.expert_slice(k)) if parent_name.endswith(".proj")
                         else (mlp.expert_slice(k), slice(None)), k)
                        for k in range(len(mlp.widths))
                    ]
            elif isinstance(parent, LoRAMLP) and leaf.startswith(("A_", "B_")):
                slices = [
                    ((parent.expert_slice(k), slice(None)) if leaf.startswith("A_")
                     else (slice(None), parent.expert_slice(k)), k)
                    for k in range(1, parent.num_experts)
                ]

            embedding = isinstance(parent, torch.nn.Embedding)
            output_head = parent_name.endswith("lm_head")
            use_muon = param.ndim == 2 and not embedding and not output_head
            for index, expert in slices:
                value = param if index is None else param[index]
                if value.numel() == 0:
                    continue
                master = torch.nn.Parameter(value.detach().to(torch.float32).clone(memory_format=torch.contiguous_format))
                self.updates.append(ParameterUpdate(param, master, index, expert))
                if use_muon:
                    muon.append(master)
                elif embedding or param.ndim < 2:
                    # Trinity explicitly exempts embeddings from decay; norms
                    # also get no decay to keep their gains stable.
                    adam_no_decay.append(master)
                else:
                    adam_decay.append(master)

        self.optimizers = [torch.optim.Muon(
            muon, lr=train.muon_lr, weight_decay=train.weight_decay,
            momentum=train.muon_momentum, nesterov=True,
            ns_steps=train.muon_ns_steps, adjust_lr_fn="original",
        )]
        adam_groups = []
        for params, decay in ((adam_decay, train.weight_decay), (adam_no_decay, 0.0)):
            if params:
                adam_groups.append({"params": params, "weight_decay": decay})
        if adam_groups:
            self.optimizers.append(torch.optim.AdamW(
                adam_groups, lr=train.lr, betas=train.adam_betas, eps=1e-8,
                fused=next(raw.parameters()).is_cuda,
            ))

    @torch.no_grad()
    def step(self, active_experts: torch.Tensor | None = None) -> None:
        """active_experts: (K,) globally reduced flags, or None for dense models."""
        active = None if active_experts is None else active_experts.tolist()
        for update in self.updates:
            grad = update.source.grad
            if grad is None or (active is not None and update.expert is not None and not active[update.expert]):
                update.master.grad = None
            else:
                update.master.grad = (grad if update.index is None else grad[update.index]).float()
        for optimizer in self.optimizers:
            optimizer.step()
        for update in self.updates:
            if update.master.grad is not None:
                destination = update.source if update.index is None else update.source[update.index]
                destination.copy_(update.master)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.model.zero_grad(set_to_none=set_to_none)
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)
