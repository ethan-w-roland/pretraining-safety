"""Trinity-style pretraining optimization, with independent routed expert updates.

Muon uses PyTorch's original shape adjustment, lr * sqrt(max(1, out / in)).
Embeddings, output heads, and scalar/vector parameters use AdamW. FP32 master
weights and optimizer states preserve small updates while model compute is bf16.
"""

from dataclasses import dataclass

import torch

from src.model.routed import RoutedTransformer
from src.run.util.config import TrainConfig
from src.run.util.distributed import get_raw_model, is_distributed


def adamw_betas(batch_tokens: int) -> tuple[float, float]:
    """Adapt Porian et al.'s Table 4 beta choices to a global token batch.

    https://arxiv.org/html/2406.19146v3 (Appendix C, Table 4)
    They keep beta1=0.9, use beta2=0.99 through batch size 192, and
    beta2=0.95 from 256 onward, at sequence length 2048. We use the first
    large-batch entry (256 * 2048 tokens) as the cutoff; this token-based
    cutoff is our adaptation of their discrete table, not a fitted law.
    Include every accumulated microbatch on every rank in batch_tokens.
    """
    if batch_tokens <= 0:
        raise ValueError("AdamW batch_tokens must be positive")
    return 0.9, 0.99 if batch_tokens < 256 * 2048 else 0.95


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
    orthogonalization stay local to an expert. Experts with no sequences in a
    step (see ``observe``) skip momentum and weight decay as well as gradient
    updates. Source gradients must already be accumulated, globally reduced by
    DDP, and clipped before ``step``.
    """

    def __init__(self, model: torch.nn.Module, train: TrainConfig):
        self.model = model
        self.updates: list[ParameterUpdate] = []
        raw = get_raw_model(model)
        routed = isinstance(raw, RoutedTransformer)

        # Routed models split each feedforward matrix into per-expert slices,
        # each its own Muon matrix; every other parameter belongs to core
        # (expert 0). Dense parameters belong to no expert and always update.
        expert_slices: dict[int, list[tuple[tuple | None, int]]] = {}
        if routed:
            for param, index, expert in raw.expert_slices():
                expert_slices.setdefault(id(param), []).append((index, expert))
        whole = [(None, 0 if routed else None)]

        # (K,) flags of the experts seen since the last step
        self.active = torch.zeros(len(raw.labels), dtype=torch.int32, device=next(raw.parameters()).device) if routed else None

        muon, adam_decay, adam_no_decay = [], [], []
        for name, param in raw.named_parameters():
            if not param.requires_grad:
                continue
            parent_name = name.rsplit(".", 1)[0]
            embedding = isinstance(raw.get_submodule(parent_name), torch.nn.Embedding)
            output_head = parent_name.endswith("lm_head")
            use_muon = param.ndim == 2 and not embedding and not output_head
            for index, expert in expert_slices.get(id(param), whole):
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
                adam_groups, lr=train.adamw_lr,
                betas=train.adamw_betas,
                eps=1e-8,
                fused=next(raw.parameters()).is_cuda,
            ))

    def observe(self, labels: torch.Tensor) -> None:
        """Mark the experts of a micro-batch's sequences, (B,) label indices, as active this step.

        No-op for dense models.
        """
        if self.active is not None:
            self.active[labels.unique()] = 1

    @torch.no_grad()
    def step(self) -> None:
        active = None
        if self.active is not None:
            if is_distributed():
                torch.distributed.all_reduce(self.active, op=torch.distributed.ReduceOp.MAX)
            active = self.active.tolist()
            self.active.zero_()
        for update in self.updates:
            grad = update.source.grad
            if grad is None or (active is not None and not active[update.expert]):
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
