"""
MoE-style gradient routing: one wide SwiGLU per layer whose hidden units are
partitioned among labels.

Each layer's feedforward is a single SwiGLU (``Qwen3MLP`` layout, no biases:
``fc_1``/``fc_2``/``proj`` are ``gate_proj``/``up_proj``/``down_proj``) of hidden width ``core_dim + aux_dim``. The combined auxiliary width budget is
divided among the ``K - 1`` auxiliary experts. Hidden units are
split into contiguous slices, slice ``k`` belonging to ``labels[k]`` (core is
slice 0, of width ``core_dim``; aux widths differ by at most one unit). Expert
``k``'s parameters are therefore rows ``k`` of ``fc_1``/``fc_2`` and columns
``k`` of ``proj``, and the whole MLP acts as one auxiliary adapter whose
parameters are masked dynamically:

- ``fwd_mask[b, t, k]`` scales expert ``k``'s hidden units at ``(b, t)``.
  Since ``proj`` is linear per slice this scales expert ``k``'s output
  contribution exactly.
- ``bck_mask[b, t, k]`` masks the gradient of expert ``k``'s rows/columns at
  ``(b, t)`` (see ``src.model.routed``).

Compared to running ``K`` separate MLPs this is three matmuls regardless of
``K`` and the same static graph for every routing pattern.
"""

from __future__ import annotations

from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import RoutedModelConfig
from src.model.routed import RoutedTransformer, adapter_dims, routed_linear, unit_owner, unit_slice


class MoEMLP(nn.Module):
    """SwiGLU whose hidden units are partitioned among experts.

    ``widths[k]`` hidden units belong to expert ``k``; ``fc_1``, ``fc_2`` and
    ``proj`` are ``Qwen3MLP``'s ``gate_proj``, ``up_proj``, ``down_proj`` with
    ``intermediate_size = sum(widths)``.
    """

    def __init__(self, embed_dim: int, widths: list[int], dtype: torch.dtype | None = None) -> None:
        super().__init__()

        hidden_dim = sum(widths)
        self.widths = list(widths)
        self.fc_1 = nn.Linear(embed_dim, hidden_dim, bias=False, dtype=dtype) # gate
        self.fc_2 = nn.Linear(embed_dim, hidden_dim, bias=False, dtype=dtype) # up
        self.proj = nn.Linear(hidden_dim, embed_dim, bias=False, dtype=dtype) # down

        self.register_buffer("unit_expert", unit_owner(widths), persistent=False) # (H,) expert of each hidden unit

    def expert_slice(self, k: int) -> slice:
        """Hidden-unit slice owned by expert ``k``."""
        return unit_slice(self.widths, k)

    def expert_params(self, k: int) -> Generator[torch.Tensor, None, None]:
        """Views of the parameters owned by expert ``k``."""
        s = self.expert_slice(k)
        yield self.fc_1.weight[s] # (width_k, E)
        yield self.fc_2.weight[s] # (width_k, E)
        yield self.proj.weight[:, s] # (E, width_k)

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

        Masks are applied as tensor ops, so the graph is the same for every
        routing pattern. A zero forward weight gives an exactly-zero output
        contribution and exactly-zero parameter gradients for that expert at
        that position.
        """
        fwd_h = fwd_mask[..., self.unit_expert] # (B, T, H) expert mask spread over hidden units
        bck_h = bck_mask[..., self.unit_expert] # (B, T, H)

        gate = routed_linear(x, self.fc_1.weight, out_mask=bck_h) # (B, T, H)
        up = routed_linear(x, self.fc_2.weight, out_mask=bck_h) # (B, T, H)
        h = F.silu(gate) * up * fwd_h
        return routed_linear(h, self.proj.weight, in_mask=bck_h) # (B, T, E)


def moe_widths(config: RoutedModelConfig, num_experts: int) -> list[int]:
    """Split the combined aux budget evenly, preserving its exact total width."""
    core_dim, aux_dim = adapter_dims(config)
    num_aux = num_experts - 1
    assert num_aux > 0, "MoE requires at least one auxiliary expert"
    assert aux_dim >= num_aux, "auxiliary width budget must give each expert at least one unit"
    width, remainder = divmod(aux_dim, num_aux)
    return [core_dim] + [width + (k < remainder) for k in range(num_aux)]


class MoETransformer(RoutedTransformer):
    """Qwen3 trunk with a slice-partitioned SwiGLU (``MoEMLP``) in every layer."""

    def __init__(
        self,
        config: RoutedModelConfig,
        labels: list[str],
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        widths = moe_widths(config, len(labels))
        super().__init__(config, labels, make_mlp=lambda: MoEMLP(config.embed_dim, widths, dtype=dtype), dtype=dtype)
        self.widths = widths # hidden units per expert
