"""
MoE-style gradient routing: one wide SwiGLU per layer whose hidden units are
partitioned among labels.

Each layer's feedforward is a single SwiGLU (``Qwen3MLP`` layout, no biases:
``fc_1``/``fc_2``/``proj`` are ``gate_proj``/``up_proj``/``down_proj``) of
hidden width ``core_dim + aux_dim``. The combined auxiliary width budget is
divided among the ``K - 1`` auxiliary experts. Hidden units are split into
contiguous slices, slice ``k`` belonging to ``labels[k]`` (core is slice 0, of
width ``core_dim``; aux widths differ by at most one unit). Expert ``k``'s
parameters are therefore rows ``k`` of ``fc_1``/``fc_2`` and columns ``k`` of
``proj``, declared as the ``owner`` of each ``RoutedLinear``'s output
(``fc_1``/``fc_2``) or input (``proj``) units:

- ``fwd_mask[b, t, k]`` scales expert ``k``'s hidden units at ``(b, t)``.
  Since ``proj`` is linear per slice this scales expert ``k``'s output
  contribution exactly.
- ``bck_mask[b, t, k]`` masks the gradient of expert ``k``'s rows/columns at
  ``(b, t)`` (applied by the routed linears, see ``src.model.routed``).

Compared to running ``K`` separate MLPs this is three matmuls regardless of
``K`` and the same static graph for every routing pattern.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import RoutedModelConfig
from src.model.routed import RoutedLinear, RoutingState, RoutedTransformer, adapter_dims, unit_owner


class MoEMLP(nn.Module):
    """SwiGLU whose hidden units are partitioned among experts.

    ``widths[k]`` hidden units belong to expert ``k``; ``fc_1``, ``fc_2`` and
    ``proj`` are ``Qwen3MLP``'s ``gate_proj``, ``up_proj``, ``down_proj`` with
    ``intermediate_size = sum(widths)``, initialised ``N(0, 0.02)`` as in HF.
    ``routing`` supplies the forward mask.
    """

    def __init__(self, embed_dim: int, widths: list[int], routing: RoutingState, dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.routing = routing
        hidden_dim = sum(widths)
        unit_expert = unit_owner(widths) # (H,) expert of each hidden unit
        self.widths = list(widths)
        self.fc_1 = RoutedLinear(embed_dim, hidden_dim, owner=unit_expert, side="out", dtype=dtype) # gate
        self.fc_2 = RoutedLinear(embed_dim, hidden_dim, owner=unit_expert, side="out", dtype=dtype) # up
        self.proj = RoutedLinear(hidden_dim, embed_dim, owner=unit_expert, side="in", dtype=dtype) # down
        for linear in (self.fc_1, self.fc_2, self.proj):
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)

        self.register_buffer("unit_expert", unit_expert, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, T, E)
        returns: (B, T, E)

        A zero forward weight gives an exactly-zero output contribution and
        exactly-zero parameter gradients for that expert at that position.
        """
        h = F.silu(self.fc_1(x)) * self.fc_2(x) # (B, T, H)
        if self.routing.fwd is not None:
            h = h * self.routing.fwd[..., self.unit_expert] # expert weights spread over hidden units
        return self.proj(h) # (B, T, E)


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
        super().__init__(config, labels, make_mlp=lambda routing: MoEMLP(config.embed_dim, widths, routing, dtype=dtype), dtype=dtype)
        self.widths = widths # hidden units per expert
