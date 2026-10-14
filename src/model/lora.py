"""
LoRA-style gradient routing: a core SwiGLU plus one big LoRA whose rank units
are partitioned among aux labels.

Each layer's feedforward is a dense SwiGLU of hidden width ``core_dim`` (the
core, expert 0) and, on each of its three linears, a single low-rank adapter
of total rank ``R = (K - 1) * rank``. The rank units are split into
contiguous slices of ``rank``, slice ``k - 1`` belonging to aux label ``k``:
expert ``k``'s parameters are rows ``k`` of each ``A`` and columns ``k`` of
each ``B``, declared as the ``owner`` of each ``A``'s output and each ``B``'s
input units. For a linear ``W`` with adapter ``(A, B)``:

    y = fwd[0] * x W^T  +  (fwd[k] * (x A^T)) B^T      summed over rank units

so with ``fwd[k] in {0, 1}`` the adapter is exactly the sum of the active
experts' rank-``rank`` LoRAs, and ``fwd[0]`` scales the core's output. As in
the pre-refactor model the adapter sits on every linear, so a fractional
weight scales each one independently (the MLP is not linear in the mask).

- ``fwd_mask[b, t, k]`` scales expert ``k``'s rank units (core: its linear
  outputs) at ``(b, t)``.
- ``bck_mask[b, t, k]`` masks the gradient of expert ``k``'s rows/columns
  (core: its dense weights) at ``(b, t)`` (applied by the routed linears, see
  ``src.model.routed``).

``rank`` is chosen so one aux expert has the same parameter count as a
``moe`` aux expert of width ``aux_dim``: ``3 * E * aux_dim`` vs
``3 * rank * (E + core_dim)``. Override with ``config.lora_rank``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import RoutedModelConfig
from src.model.routed import RoutedLinear, RoutingState, RoutedTransformer, adapter_dims, unit_owner


def calc_lora_rank(embed_dim: int, core_dim: int, aux_dim: float) -> int:
    """Rank giving one aux expert the parameter count of a width-``aux_dim`` SwiGLU expert.

    MoE aux expert: ``3 * E * aux_dim``. LoRA aux expert on ``fc_1``, ``fc_2``
    (``E -> core_dim``) and ``proj`` (``core_dim -> E``): ``3 * r * (E + core_dim)``.
    """
    return max(1, round(embed_dim * aux_dim / (embed_dim + core_dim)))


class LoRAMLP(nn.Module):
    """Core SwiGLU of width ``core_dim`` with a rank-``R`` LoRA on each linear.

    ``R = (num_experts - 1) * rank``; rank slice ``k - 1`` belongs to aux expert
    ``k``. ``A_i`` maps ``in -> R`` and ``B_i`` maps ``R -> out``. The core
    linears are plain ``nn.Linear`` (``make_routed`` assigns them to core).
    Init: core and ``A ~ N(0, 0.02)``, ``B = 0``, so the adapter starts as the
    identity on the core. ``routing`` supplies the forward mask.
    """

    def __init__(self, embed_dim: int, core_dim: int, num_experts: int, rank: int, routing: RoutingState, dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.routing = routing
        self.rank = rank
        self.num_experts = num_experts
        R = (num_experts - 1) * rank
        unit_expert = unit_owner([rank] * (num_experts - 1), first_expert=1) # (R,) expert of each rank unit

        def lora(in_dim: int, out_dim: int) -> tuple[RoutedLinear, RoutedLinear]:
            A = RoutedLinear(in_dim, R, owner=unit_expert, side="out", dtype=dtype) # (R, in)
            B = RoutedLinear(R, out_dim, owner=unit_expert, side="in", dtype=dtype) # (out, R)
            nn.init.normal_(A.weight, mean=0.0, std=0.02)
            nn.init.zeros_(B.weight)
            return A, B

        self.fc_1 = nn.Linear(embed_dim, core_dim, bias=False, dtype=dtype) # gate
        self.fc_2 = nn.Linear(embed_dim, core_dim, bias=False, dtype=dtype) # up
        self.proj = nn.Linear(core_dim, embed_dim, bias=False, dtype=dtype) # down
        for linear in (self.fc_1, self.fc_2, self.proj):
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
        self.A_1, self.B_1 = lora(embed_dim, core_dim)
        self.A_2, self.B_2 = lora(embed_dim, core_dim)
        self.A_3, self.B_3 = lora(core_dim, embed_dim)

        self.register_buffer("unit_expert", unit_expert, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, T, E)
        returns: (B, T, E)

        A zero forward weight gives an exactly-zero contribution and
        exactly-zero parameter gradients for that expert at that position.
        """
        fwd_core = fwd_r = 1.0
        if self.routing.fwd is not None:
            fwd_core = self.routing.fwd[..., :1] # (B, T, 1) core forward weight
            fwd_r = self.routing.fwd[..., self.unit_expert] # (B, T, R) expert weights spread over rank units

        def adapted(x, linear, A, B):
            """``fwd_core * (x W^T) + (fwd_r * (x A^T)) B^T``"""
            return linear(x) * fwd_core + B(A(x) * fwd_r)

        gate = adapted(x, self.fc_1, self.A_1, self.B_1) # (B, T, H)
        up = adapted(x, self.fc_2, self.A_2, self.B_2) # (B, T, H)
        h = F.silu(gate) * up
        return adapted(h, self.proj, self.A_3, self.B_3) # (B, T, E)


class LoRATransformer(RoutedTransformer):
    """Qwen3 trunk with a core SwiGLU + partitioned LoRA (``LoRAMLP``) in every layer."""

    def __init__(
        self,
        config: RoutedModelConfig,
        labels: list[str],
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        core_dim, aux_dim = adapter_dims(config)
        # aux_param_prc budgets all auxiliary experts together. Integer ranks
        # approximate that budget; unlike MoE slices they cannot match it exactly.
        rank = config.lora_rank if config.lora_rank is not None else calc_lora_rank(config.embed_dim, core_dim, aux_dim / (len(labels) - 1))
        super().__init__(
            config, labels,
            make_mlp=lambda routing: LoRAMLP(config.embed_dim, core_dim, len(labels), rank, routing, dtype=dtype),
            dtype=dtype,
        )
        self.core_dim = core_dim
        self.rank = rank # rank units per aux expert
