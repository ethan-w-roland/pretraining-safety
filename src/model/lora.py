"""
LoRA-style gradient routing: a core SwiGLU plus one big LoRA whose rank units
are partitioned among aux labels.

Each layer's feedforward is a dense SwiGLU of hidden width ``core_dim`` (the
core, expert 0) and, on each of its three linears, a single low-rank adapter
of total rank ``R = (K - 1) * rank``. The rank units are split into
contiguous slices of ``rank``, slice ``k - 1`` belonging to aux label ``k``:
expert ``k``'s parameters are rows ``k`` of each ``A`` and columns ``k`` of
each ``B``. For a linear ``W`` with adapter ``(A, B)``:

    y = fwd[0] * x W^T  +  (fwd[k] * (x A^T)) B^T      summed over rank units

so with ``fwd[k] in {0, 1}`` the adapter is exactly the sum of the active
experts' rank-``rank`` LoRAs, and ``fwd[0]`` scales the core's output. As in
the pre-refactor model the adapter sits on every linear, so a fractional
weight scales each one independently (the MLP is not linear in the mask).

- ``fwd_mask[b, t, k]`` scales expert ``k``'s rank units (core: its linear
  outputs) at ``(b, t)``.
- ``bck_mask[b, t, k]`` masks the gradient of expert ``k``'s rows/columns
  (core: its dense weights) at ``(b, t)`` (see ``src.model.routed``).

``rank`` is chosen so one aux expert has the same parameter count as a
``moe`` aux expert of width ``aux_dim``: ``3 * E * aux_dim`` vs
``3 * rank * (E + core_dim)``. Override with ``config.lora_rank``.
"""

from __future__ import annotations

from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import RoutedModelConfig
from src.model.routed import RoutedTransformer, adapter_dims, routed_linear, unit_owner, unit_slice


def calc_lora_rank(embed_dim: int, core_dim: int, aux_dim: float) -> int:
    """Rank giving one aux expert the parameter count of a width-``aux_dim`` SwiGLU expert.

    MoE aux expert: ``3 * E * aux_dim``. LoRA aux expert on ``fc_1``, ``fc_2``
    (``E -> core_dim``) and ``proj`` (``core_dim -> E``): ``3 * r * (E + core_dim)``.
    """
    return max(1, round(embed_dim * aux_dim / (embed_dim + core_dim)))


class LoRAMLP(nn.Module):
    """Core SwiGLU of width ``core_dim`` with a rank-``R`` LoRA on each linear.

    ``R = (num_experts - 1) * rank``; rank slice ``k - 1`` belongs to aux expert
    ``k``. ``A`` matrices are ``(R, in)`` and ``B`` matrices ``(out, R)`` so
    both are applied with ``routed_linear``. ``A ~ N(0, 0.02)``, ``B = 0`` so
    the adapter starts as the identity on the core.
    """

    def __init__(self, embed_dim: int, core_dim: int, num_experts: int, rank: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.rank = rank
        self.num_experts = num_experts
        R = (num_experts - 1) * rank

        self.fc_1 = nn.Linear(embed_dim, core_dim, bias=False, dtype=dtype) # gate
        self.fc_2 = nn.Linear(embed_dim, core_dim, bias=False, dtype=dtype) # up
        self.proj = nn.Linear(core_dim, embed_dim, bias=False, dtype=dtype) # down

        self.A_1 = nn.Parameter(torch.empty(R, embed_dim, dtype=dtype)) # (R, E)
        self.B_1 = nn.Parameter(torch.zeros(core_dim, R, dtype=dtype)) # (H, R)
        self.A_2 = nn.Parameter(torch.empty(R, embed_dim, dtype=dtype)) # (R, E)
        self.B_2 = nn.Parameter(torch.zeros(core_dim, R, dtype=dtype)) # (H, R)
        self.A_3 = nn.Parameter(torch.empty(R, core_dim, dtype=dtype)) # (R, H)
        self.B_3 = nn.Parameter(torch.zeros(embed_dim, R, dtype=dtype)) # (E, R)
        for A in (self.A_1, self.A_2, self.A_3):
            nn.init.normal_(A, mean=0.0, std=0.02)

        # expert index of each rank unit, (R,); aux experts start at 1
        self.register_buffer("unit_expert", unit_owner([rank] * (num_experts - 1), first_expert=1), persistent=False)

    def expert_slice(self, k: int) -> slice:
        """Rank-unit slice owned by aux expert ``k >= 1``."""
        assert k >= 1, "core (expert 0) owns the dense weights, not rank units"
        return unit_slice([self.rank] * (self.num_experts - 1), k - 1)

    def expert_params(self, k: int) -> Generator[torch.Tensor, None, None]:
        """Parameters owned by expert ``k``: dense weights for core, ``A``/``B`` slice views otherwise."""
        if k == 0:
            yield self.fc_1.weight # (H, E)
            yield self.fc_2.weight # (H, E)
            yield self.proj.weight # (E, H)
            return
        s = self.expert_slice(k)
        for A, B in ((self.A_1, self.B_1), (self.A_2, self.B_2), (self.A_3, self.B_3)):
            yield A[s] # (rank, in)
            yield B[:, s] # (out, rank)

    @staticmethod
    def _adapted(
        x: torch.Tensor,
        weight: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        fwd_core: torch.Tensor,
        bck_core: torch.Tensor,
        fwd_r: torch.Tensor,
        bck_r: torch.Tensor,
    ) -> torch.Tensor:
        """``fwd_core * (x W^T) + (fwd_r * (x A^T)) B^T`` with routed gradients.

        x:        (B, T, in)
        weight:   (out, in) core weight
        A:        (R, in)
        B:        (out, R)
        fwd_core: (B, T, 1) core forward weight
        bck_core: (B, T, 1) core gradient weight
        fwd_r:    (B, T, R) forward weight of each rank unit's expert
        bck_r:    (B, T, R) gradient weight of each rank unit's expert
        returns:  (B, T, out)
        """
        y = routed_linear(x, weight, out_mask=bck_core) * fwd_core # (B, T, out)
        z = routed_linear(x, A, out_mask=bck_r) * fwd_r # (B, T, R)
        return y + routed_linear(z, B, in_mask=bck_r) # (B, T, out)

    def forward(
        self,
        x: torch.Tensor,
        fwd_mask: torch.Tensor,
        bck_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:        (B, T, E)
        fwd_mask: (B, T, K) per-position forward weight of each expert, in x's dtype
        bck_mask: (B, T, K) per-position gradient weight of each expert, in x's dtype
        returns:  (B, T, E)

        A zero forward weight gives an exactly-zero contribution and
        exactly-zero parameter gradients for that expert at that position.
        """
        fwd_c, bck_c = fwd_mask[..., :1], bck_mask[..., :1] # (B, T, 1) core
        fwd_r = fwd_mask[..., self.unit_expert] # (B, T, R) expert mask spread over rank units
        bck_r = bck_mask[..., self.unit_expert] # (B, T, R)
        masks = (fwd_c, bck_c, fwd_r, bck_r)

        gate = self._adapted(x, self.fc_1.weight, self.A_1, self.B_1, *masks) # (B, T, H)
        up = self._adapted(x, self.fc_2.weight, self.A_2, self.B_2, *masks) # (B, T, H)
        h = F.silu(gate) * up
        return self._adapted(h, self.proj.weight, self.A_3, self.B_3, *masks) # (B, T, E)


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
            make_mlp=lambda: LoRAMLP(config.embed_dim, core_dim, len(labels), rank, dtype=dtype),
            dtype=dtype,
        )
        self.core_dim = core_dim
        self.rank = rank # rank units per aux expert
