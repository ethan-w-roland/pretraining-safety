"""
Shared machinery for gradient-routed transformers.

Two concrete models build on this module. Both are the stock Qwen3 model of
``src.model.base`` (``transformers.Qwen3ForCausalLM``) with every layer's MLP
swapped for one adapter whose parameters are owned by labels:

- ``src.model.moe``:  one wide SwiGLU whose hidden units are partitioned into
  per-label slices (core slice + one slice per aux label).
- ``src.model.lora``: a core SwiGLU plus one big LoRA on each of its linears,
  whose rank units are partitioned into per-aux-label slices.

Masks have ``K = len(labels)`` columns, indexed like ``labels`` (core is 0):

- ``fwd_mask[b, t, k]`` scales expert ``k``'s contribution at position
  ``(b, t)``. Boolean (multi-hot) in training; fractional in ``[0, 1]`` for
  titration. Only the adapters' aux units have forward weights (applied inside
  ``MoEMLP`` / ``LoRAMLP``); the trunk always runs in full.
- ``bck_mask[b, t, k]`` routes the parameter gradient of expert ``k`` at
  position ``(b, t)``: 0 is exactly "frozen" at that position, 1 is exactly
  "trainable". Activations and input gradients always flow, so experts in
  earlier layers train through later ones.

Masks of shape ``(K,)`` or ``(B, K)`` are broadcast to ``(B, T, K)``.

Ownership
---------
Routing lives in the parameterised leaf modules, not in the model code:
``RoutedLinear``, ``RoutedRMSNorm`` and ``RoutedEmbedding`` each carry an
``owner`` vector, the expert of each of their units:

- ``owner`` of length 1: the whole leaf belongs to one expert. Every stock
  leaf of the Qwen3 trunk (attention projections, norms, embedding, output
  head) is owned by core this way, so ``make_routed`` swaps them in place and
  the HF attention, decoder layer and model forward run unchanged.
- ``owner`` of length ``U``: the leaf's output (``side="out"``) or input
  (``side="in"``) units are partitioned among experts, e.g. the MoE's hidden
  units. The adapters declare these themselves.

The same owners tell the optimizer how to split parameters into per-expert
matrices (``RoutedTransformer.expert_slices``).

The masks of the forward in progress live on one ``RoutingState``, set by
``RoutedTransformer.forward`` around the call into the HF model and shared by
the routed modules (for backward masks) and the adapters (for forward masks).

Gradient routing
----------------
The pre-refactor models froze an expert for a whole forward pass by running it
through ``functional_call`` with detached parameters. That is all-or-nothing
per forward pass, since a parameter is shared by every token in the batch.
The routed modules make the decision per position (and per unit). For
``y = x W^T`` with unit mask ``m = bck_mask[..., owner]`` on the output side:

    dL/dx = g W                                   unchanged
    dL/dW = sum_{b,t} (g[b,t] * m[b,t])^T x[b,t]

so a masked position/unit is exactly "parameters frozen" there. The op costs
one gather and one elementwise multiply in backward, and builds a single static
graph regardless of mask contents, so ``torch.compile`` does not recompile per
routing pattern. Masks are indexed by the position at which the parameter is
*applied*, not the position of the loss it ultimately serves.
"""

from __future__ import annotations

from typing import Callable, Generator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from src.model.config import RoutedModelConfig
from src.model.base import QwenTransformer


# --------------------------------------------------------------------------- #
# routed ops                                                                  #
# --------------------------------------------------------------------------- #


class _RoutedLinear(torch.autograd.Function):
    """``F.linear`` whose weight/bias gradients are masked per position and per unit.

    Shapes use ``*`` for any leading dims (``(B, T)`` in the transformer).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        bck_mask: torch.Tensor,
        owner: torch.Tensor,
        out_side: bool,
    ):
        """
        x:        (*, in)
        weight:   (out, in)
        bias:     (out,) or None
        bck_mask: (*, K) gradient weight of each expert
        owner:    (U,) expert of each output unit (out_side) or input unit;
                  U = 1 means one expert owns the whole weight
        returns:  (*, out)
        """
        # Save the (*, K) expert mask, not the (*, U) unit mask: every leaf
        # shares the same expert mask, so this costs no memory per leaf.
        ctx.save_for_backward(x, weight, bck_mask, owner)
        ctx.out_side = out_side
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        """
        g:       (*, out) gradient w.r.t. the output
        returns: (gx, gw, gb, None, None, None)
                 gx: (*, in)   = g @ weight, unmasked
                 gw: (out, in) = sum over positions of (g * m_out)^T (x * m_in)
                 gb: (out,)    = sum over positions of g * m_out
                 where the unit mask m = bck_mask[..., owner] is m_out or m_in
        """
        # Forward computed y = x @ weight.T + bias without using the mask.
        # g is the incoming dL/dy, shaped (*, out); * is usually (B, T).
        # The mask customizes parameter updates, rather than the forward value.
        # Bias values are unnecessary: dL/db depends on g, not on the bias.
        x, weight, bck_mask, owner = ctx.saved_tensors

        # Flags correspond to forward's arguments, in order. Compute only
        # required gradients; the mask and owners are not differentiated.
        need_x, need_w, need_b = ctx.needs_input_grad[:3]

        # Leave unneeded gradients as None. A required gradient can still be a
        # tensor of zeros when every contributing position is masked out.
        gx = gw = gb = None
        if need_x:
            # Ordinary input gradient: (*, out) @ (out, in) -> (*, in).
            # Deliberately unmasked: freezing this layer's parameters at a
            # token must still let that token train upstream layers through
            # the weights used in this forward pass. Any forward masking
            # elsewhere already affects the arriving g.
            gx = g @ weight

        if need_w or need_b:
            # Unit mask (*, U): the gradient weight of each unit's owner at
            # each position. U = 1 broadcasts one factor over all units; zero
            # blocks a contribution, fractional values scale it.
            mask = bck_mask[..., owner]

            # Output-side units select weight ROWS (and bias elements);
            # input-side units select weight COLUMNS.
            gm = g * mask.to(g.dtype) if ctx.out_side else g
            xm = x if ctx.out_side else x * mask.to(x.dtype)

            # Merge leading dimensions into N positions (N = B*T normally):
            # (*, out) -> (N, out). Weight/bias are shared across positions,
            # so their gradients must SUM contributions across this dimension.
            gm = gm.reshape(-1, g.shape[-1])
            if need_w:
                # (out, N) @ (N, in) -> (out, in), matching weight's shape.
                # Each position contributes an outer product of its (masked)
                # output gradient and (masked) input:
                # gw[o, i] = sum_p g[p, o] * x[p, i] * m[p, o or i].
                gw = gm.t() @ xm.reshape(-1, x.shape[-1])
            if need_b:
                # gb[o] = sum_p g[p, o] * m[p, o]; summing (N, out) over N
                # gives (out,). Input-side routing has no bias (RoutedLinear).
                gb = gm.sum(0)

        # Autograd requires one result per forward argument, in the same order.
        return gx, gw, gb, None, None, None


class _ScaleGradient(torch.autograd.Function):
    """Identity forward; backward scales the gradient by ``mask`` per position.

    Applied to a per-position view of a parameter (an embedding lookup, or a
    norm gain expanded over positions), this masks that parameter's gradient
    contribution at each position.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, mask: torch.Tensor):
        """x: (B, T, ...), mask: broadcastable to x -> unchanged x."""
        ctx.save_for_backward(mask)
        return x

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        """g: (B, T, ...) -> g * mask, no mask grad."""
        (mask,) = ctx.saved_tensors
        return g * mask, None


def expand_mask(
    mask: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    num_experts: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Broadcast an expert mask to ``(B, T, K)`` in ``dtype``.

    mask:    None            -> all ones
             (K,)            -> same weights at every position
             (B, K)          -> per sequence
             (B, T, K)       -> per token
    returns: (B, T, K) expanded view (no memory materialized for the broadcast cases)
    """
    if mask is None:
        mask = torch.ones(num_experts, dtype=dtype, device=device)
    mask = mask.to(dtype=dtype, device=device)

    assert mask.shape[-1] == num_experts, f"mask has {mask.shape[-1]} experts, model has {num_experts}"
    if mask.dim() == 1:
        mask = mask.view(1, 1, num_experts)
    elif mask.dim() == 2:
        mask = mask.view(batch_size, 1, num_experts)
    else:
        assert mask.dim() == 3, f"mask must have shape (K,), (B, K) or (B, T, K), got {tuple(mask.shape)}"

    return mask.expand(batch_size, seq_len, num_experts)


def adapter_dims(config: RoutedModelConfig, align: int = 64) -> Tuple[int, int]:
    """``(core_dim, aux_dim)``: core width and combined auxiliary width budget.

    ``core_dim = core_param_prc * mlp_dim`` and ``aux_dim = aux_param_prc * mlp_dim``,
    Core width is rounded to ``align``. When shares sum to one, auxiliary
    width is the exact remainder of ``mlp_dim`` so total width is preserved.
    Otherwise auxiliary width is independently rounded to ``align``.
    """
    core_dim = round(config.core_param_prc * config.mlp_dim / align) * align
    if abs(config.core_param_prc + config.aux_param_prc - 1.0) < 1e-9:
        aux_dim = config.mlp_dim - core_dim
    else:
        aux_dim = round(config.aux_param_prc * config.mlp_dim / align) * align
    assert core_dim > 0 and aux_dim > 0, f"core_dim={core_dim}, aux_dim={aux_dim} must be positive"
    return core_dim, aux_dim


def unit_owner(widths: list[int], first_expert: int = 0) -> torch.Tensor:
    """Expert index of every unit, (sum(widths),), for units laid out as ``widths`` in order.

    Unit ``u`` in slice ``k`` maps to ``first_expert + k``. Integer tensor, so
    ``.to(dtype)`` on the owning module leaves it alone when registered as a buffer.
    """
    owner = torch.repeat_interleave(torch.arange(len(widths)), torch.tensor(widths, dtype=torch.long))
    return owner + first_expert


# --------------------------------------------------------------------------- #
# routed modules                                                               #
# --------------------------------------------------------------------------- #


class RoutingState:
    """Expert masks of the routed forward in progress, shared by a model's routed modules.

    fwd: (B, T, K) forward weight of each expert, or None outside a routed forward
    bck: (B, T, K) gradient weight of each expert, or None outside a routed forward
    """

    def __init__(self) -> None:
        self.fwd: Optional[torch.Tensor] = None
        self.bck: Optional[torch.Tensor] = None


class RoutedModule:
    """Mixin for a leaf module (one with parameters and no submodules: a linear,
    norm or embedding) whose parameter gradients are routed by owner.

    owner:   (U,) expert of each unit; U = 1 means one expert owns the whole leaf
    side:    ``"out"`` or ``"in"``, the weight dimension the units index (U > 1)
    routing: the model's ``RoutingState``, attached by ``make_routed``. Outside
             a routed forward its masks are None and the leaf behaves exactly
             like its base class.
    """

    routing: Optional[RoutingState] = None

    def set_owner(self, owner: torch.Tensor, side: str = "out") -> None:
        assert side in ("out", "in"), f"side must be 'out' or 'in', got {side!r}"
        if owner.numel() > 1:
            assert isinstance(self, nn.Linear), "only linears split their units among experts"
            dim = self.weight.shape[0 if side == "out" else 1]
            assert owner.numel() == dim, f"{owner.numel()} owners for {dim} {side}put units"
            assert side == "out" or self.bias is None, "input units have no bias entries to route"
        self.register_buffer("owner", owner.long(), persistent=False)
        self.side = side

    @property
    def bck_mask(self) -> Optional[torch.Tensor]:
        """(B, T, K) gradient weight of each expert, or None outside a routed forward."""
        return None if self.routing is None else self.routing.bck

    def expert_slices(self) -> Generator[tuple[nn.Parameter, tuple | None, int], None, None]:
        """``(param, index, k)``: ``param[index]`` (all of ``param`` if None) is owned by expert ``k``."""
        owner = self.owner.tolist()
        params = list(self.named_parameters(recurse=False))
        if len(owner) == 1:
            for _, param in params:
                yield param, None, owner[0]
            return
        for k in sorted(set(owner)):
            units = [u for u, o in enumerate(owner) if o == k]
            s = slice(units[0], units[-1] + 1)
            assert len(units) == s.stop - s.start, f"expert {k}'s units must be contiguous"
            for name, param in params:
                if name == "bias": # one entry per output unit
                    yield param, (s,), k
                else:
                    yield param, ((s, slice(None)) if self.side == "out" else (slice(None), s)), k


class RoutedLinear(RoutedModule, nn.Linear):
    """``nn.Linear`` whose parameter gradients are routed by ``owner``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        owner: torch.Tensor,
        side: str = "out",
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.set_owner(owner, side)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (*, in) -> (*, out)"""
        if self.bck_mask is None:
            return super().forward(x)
        return _RoutedLinear.apply(x, self.weight, self.bias, self.bck_mask, self.owner, self.side == "out")


class RoutedRMSNorm(RoutedModule, Qwen3RMSNorm):
    """``Qwen3RMSNorm`` whose gain gradient is routed by its owner."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, ..., D) -> same shape. Matches HF's fp32 normalization exactly."""
        if self.bck_mask is None:
            return super().forward(x)
        normalized = x.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = (normalized * torch.rsqrt(variance + self.variance_epsilon)).to(x.dtype)
        # Expand the gain over positions and mask its gradient there; the
        # input gradient (g * weight) is untouched, so earlier layers still train.
        mask = self.bck_mask[..., self.owner] # (B, T, 1)
        mask = mask.reshape(*mask.shape[:2], *([1] * (x.ndim - 2))) # (B, T, 1, ...): QK norms have a heads axis
        return normalized * _ScaleGradient.apply(self.weight.expand_as(normalized), mask)


class RoutedEmbedding(RoutedModule, nn.Embedding):
    """``nn.Embedding`` whose weight gradient is routed by its owner."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) -> (B, T, E)"""
        out = super().forward(tokens)
        if self.bck_mask is None:
            return out
        # The lookup has no differentiable input to preserve: mask its output
        # gradient to route the weight's contributions by position.
        return _ScaleGradient.apply(out, self.bck_mask[..., self.owner])


ROUTED_MODULES = {nn.Linear: RoutedLinear, nn.Embedding: RoutedEmbedding, Qwen3RMSNorm: RoutedRMSNorm}


def make_routed(model: nn.Module, routing: RoutingState) -> None:
    """Route every parameterised leaf of ``model`` through ``routing``'s masks.

    Stock ``nn.Linear``, ``nn.Embedding`` and ``Qwen3RMSNorm`` leaves become
    their routed versions owned by core (in place, by swapping ``__class__``,
    so ``state_dict``, ``deepcopy`` and ``torch.compile`` see ordinary
    modules). Leaves that are already routed (built by the adapters, which
    declare their own owners) are only attached to ``routing``.
    """

    core = torch.zeros(1, dtype=torch.long)
    for module in model.modules():
        if type(module) in ROUTED_MODULES:
            module.__class__ = ROUTED_MODULES[type(module)]
            module.set_owner(core)
        if isinstance(module, RoutedModule):
            module.routing = routing

    # An unowned parameter would train on every sequence, unmasked.
    owned = {id(p) for m in model.modules() if isinstance(m, RoutedModule) for p in m.parameters(recurse=False)}
    unowned = [name for name, p in model.named_parameters() if id(p) not in owned]
    assert not unowned, f"parameters outside any routed module: {unowned}"


# --------------------------------------------------------------------------- #
# model                                                                       #
# --------------------------------------------------------------------------- #


class RoutedTransformer(QwenTransformer):
    """Qwen3 model with a routed feedforward in every layer.

    Subclasses compute their adapter geometry from ``config`` and pass a
    ``make_mlp(routing)`` factory producing one feedforward module per layer:
    ``(B, T, E) -> (B, T, E)``, applying ``routing.fwd`` to its aux units and
    declaring owners on its ``RoutedLinear``s. ``labels[0]`` must be ``"core"``.
    """

    def __init__(
        self,
        config: RoutedModelConfig,
        labels: list[str],
        make_mlp: Callable[[RoutingState], nn.Module],
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:

        super().__init__(config, dtype=dtype)
        assert labels[0] == "core", "first label must be core"
        self.labels = labels

        self.routing = RoutingState()
        for layer in self.hf.model.layers:
            layer.mlp = make_mlp(self.routing)
        make_routed(self.hf, self.routing)

    def expert_slices(self) -> Generator[tuple[nn.Parameter, tuple | None, int], None, None]:
        """``(param, index, k)`` for every parameter: ``param[index]`` (all of
        ``param`` if None) is owned by expert ``k``. Read off the owners of the
        routed modules; a tied weight is listed once.
        """
        seen = set()
        for module in self.modules():
            if not isinstance(module, RoutedModule):
                continue
            for param, index, k in module.expert_slices():
                if id(param) not in seen:
                    yield param, index, k
            seen.update(id(p) for p in module.parameters(recurse=False))

    def get_params(self, label: str) -> Generator[torch.Tensor, None, None]:
        """Yield the tensors owned by ``label``.

        For ``"core"`` these are the whole trunk parameters (embed, norms,
        attention, unembed if untied) plus its feedforward parameters; for an
        aux label just its feedforward parameters. Adapter tensors are
        generally *views* into shared weights, so this is for counting and
        inspection, not for building optimizers (use ``parameters()``).
        """

        assert label in self.labels, f"label {label} not found in {self.labels}"
        k = self.labels.index(label)
        for param, index, owner in self.expert_slices():
            if owner == k:
                yield param if index is None else param[index]

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        fwd_mask: Optional[torch.Tensor] = None,
        bck_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        tokens:   (B, T) token ids
        targets:  (B, T) next-token ids, -1000 = ignored; None for no loss
        fwd_mask: (K,), (B, K) or (B, T, K) forward weight per expert; None = all experts on
        bck_mask: (K,), (B, K) or (B, T, K) gradient weight per expert; None = all experts train
                  column 0 controls the trunk AND core MLP at every position
        returns:  (logits (B, T, V), loss scalar or None)
        """
        B, T = tokens.shape
        K = len(self.labels)
        dtype = self.hf.model.embed_tokens.weight.dtype
        self.routing.fwd = expand_mask(fwd_mask, B, T, K, dtype, tokens.device) # (B, T, K)
        self.routing.bck = expand_mask(bck_mask, B, T, K, dtype, tokens.device) # (B, T, K)
        try:
            logits = self.hf(input_ids=tokens).logits # (B, T, V)
        finally:
            self.routing.fwd = self.routing.bck = None
        loss = self.loss_fn(logits, targets) if targets is not None else None
        return logits, loss
