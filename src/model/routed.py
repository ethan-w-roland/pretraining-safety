"""
Shared machinery for gradient-routed transformers.

Two concrete models build on this module. Both are the Qwen3 trunk of
``src.model.base`` (``transformers.Qwen3ForCausalLM``) with every layer's MLP
swapped for one adapter whose parameters are owned by labels and masked
dynamically per position:

- ``src.model.moe``:  one wide SwiGLU whose hidden units are partitioned into
  per-label slices (core slice + one slice per aux label).
- ``src.model.lora``: a core SwiGLU plus one big LoRA on each of its linears,
  whose rank units are partitioned into per-aux-label slices.

Masks have ``K = len(labels)`` columns, indexed like ``labels`` (core is 0):

- ``fwd_mask[b, t, k]`` scales expert ``k``'s contribution at position
  ``(b, t)``. Boolean (multi-hot) in training; fractional in ``[0, 1]`` for
  titration. A zero contributes nothing to the forward pass or to any
  gradient.
- ``bck_mask[b, t, k]`` routes the parameter gradient of expert ``k`` at
  position ``(b, t)``: 0 is exactly "frozen" at that position, 1 is exactly
  "trainable". Activations and input gradients always flow, so experts in
  earlier layers train through later ones.

Expert 0 owns both the shared trunk and the core MLP slice. Its backward mask
controls all of those parameters together; trunk freezing has no separate flag.
It can vary across sequences and tokens within a single invocation. Every core
operation masks its parameter-gradient contributions at the position where it
is applied, retaining input gradients while auxiliary MLP slices train.
Masks of shape ``(K,)`` or ``(B, K)`` are broadcast to ``(B, T, K)``.

Gradient routing
----------------
The pre-refactor models froze an expert for a whole forward pass by running it
through ``functional_call`` with detached parameters. That is all-or-nothing
per forward pass, since a parameter is shared by every token in the batch.
``routed_linear`` makes the decision per position (and per unit). For
``y = x W^T`` with an output-side mask ``m_out`` and input-side mask ``m_in``:

    dL/dx = g W                                   unchanged
    dL/dW = sum_{b,t} (g[b,t] * m_out[b,t])^T (x[b,t] * m_in[b,t])

so a masked position/unit is exactly "parameters frozen" there. The op costs
one extra elementwise multiply in backward and builds a single static graph
regardless of mask contents, so ``torch.compile`` does not recompile per
routing pattern. Masks are indexed by the position at which the parameter is
*applied*, not the position of the loss it ultimately serves.
"""

from __future__ import annotations

from typing import Callable, Generator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS, Qwen3Attention, Qwen3DecoderLayer,
    apply_rotary_pos_emb, eager_attention_forward,
)

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
        out_mask: Optional[torch.Tensor],
        in_mask: Optional[torch.Tensor],
    ):
        """
        x:        (*, in)
        weight:   (out, in)
        bias:     (out,) or None
        out_mask: (*, out) or (*, 1) gradient weight per output unit, or None
        in_mask:  (*, in) or (*, 1) gradient weight per input unit, or None
        returns:  (*, out)
        """
        ctx.save_for_backward(x, weight, out_mask, in_mask)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        """
        g:       (*, out) gradient w.r.t. the output
        returns: (gx, gw, gb, None, None)
                 gx: (*, in)   = g @ weight, unmasked
                 gw: (out, in) = sum over positions of (g * out_mask)^T (x * in_mask)
                 gb: (out,)    = sum over positions of g * out_mask
        """
        # Forward computed y = x @ weight.T + bias without using these masks.
        # g is the incoming dL/dy, shaped (*, out); * is usually (B, T).
        # The masks customize parameter updates, rather than the forward value.
        # Recover the tensors saved by forward. Bias values are unnecessary:
        # dL/db depends on g, not on the current bias.
        x, weight, out_mask, in_mask = ctx.saved_tensors

        # Flags correspond to forward's arguments, in order:
        # (x, weight, bias, out_mask, in_mask). Compute only required gradients.
        # Mask flags are ignored because we do not differentiate routing masks.
        need_x, need_w, need_b, _, _ = ctx.needs_input_grad

        # Leave unneeded gradients as None. A required gradient can still be a
        # tensor of zeros when every contributing position is masked out.
        gx = gw = gb = None
        if need_x:
            # Ordinary input gradient: (*, out) @ (out, in) -> (*, in).
            # gx[..., i] = sum_o g[..., o] * weight[o, i].
            # Deliberately omit backward masks: freezing this layer's parameter
            # contribution at a token must still let that token train upstream
            # layers through the weights used in this forward pass.
            # Any forward masking elsewhere already affects the arriving g.
            gx = g @ weight

        if need_w or need_b:
            # Output-side masking is used ONLY for weight/bias gradients.
            # At each position, out_mask[..., o] scales that position's
            # contribution to weight row o and bias element o. A mask shaped
            # (*, 1) broadcasts the same factor over all output units.
            # None means a factor of 1; zero blocks a contribution; fractional
            # values scale it. Cast the mask to the incoming gradient's dtype.
            gm = g if out_mask is None else g * out_mask.to(g.dtype)

            # Merge leading dimensions into N positions (N = B*T normally):
            # (*, out) -> (N, out). Weight/bias are shared across positions,
            # so their gradients must SUM contributions across this dimension.
            gm = gm.reshape(-1, g.shape[-1])
            if need_w:
                # Input-side masking selects weight COLUMNS: in_mask[..., i]
                # scales the contribution to column i at each position.
                # This creates a temporary for gw; it does not modify x or gx.
                # A (*, 1) mask broadcasts over input units; None means all on.
                xm = x if in_mask is None else x * in_mask.to(x.dtype)

                # (out, N) @ (N, in) -> (out, in), matching weight's shape.
                # Each position contributes an outer product of its masked
                # output gradient and masked input; matmul sums these products:
                # gw[o, i] = sum_p g[p, o] * x[p, i]
                #                   * out_mask[p, o] * in_mask[p, i].
                # Here p indexes flattened positions; omitted masks equal 1.
                # If both masks are supplied, their factors MULTIPLY (e.g.
                # 0.5 on each side gives 0.25 for that weight contribution).
                gw = gm.t() @ xm.reshape(-1, x.shape[-1])
            if need_b:
                # Bias is added directly to each output, independently of x:
                # gb[o] = sum_p g[p, o] * out_mask[p, o].
                # Only the output mask applies; summing (N, out) over N gives
                # (out,). Bias-free linears do not need this branch.
                gb = gm.sum(0)

        # Autograd requires one result per forward argument, in the same order.
        # The last two Nones explicitly provide no gradient for routing masks.
        return gx, gw, gb, None, None


def routed_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_mask: Optional[torch.Tensor] = None,
    in_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``F.linear`` with parameter gradients routed by per-position/per-unit masks.

    x:        (*, in)
    weight:   (out, in)
    bias:     (out,) or None
    out_mask: (*, out) or (*, 1) gradient weight per output unit, or None
    in_mask:  (*, in) or (*, 1) gradient weight per input unit, or None
    returns:  (*, out)

    Both masks ``None`` is plain ``F.linear``.
    """
    if out_mask is None and in_mask is None:
        return F.linear(x, weight, bias)
    return _RoutedLinear.apply(x, weight, bias, out_mask, in_mask)


class _ScaleGradient(torch.autograd.Function):
    """Identity forward; scale backward for embedding parameter contributions."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, mask: torch.Tensor):
        """x: (B, T, E), mask: (B, T, 1) -> unchanged (B, T, E)."""
        ctx.save_for_backward(mask)
        return x

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        """g: (B, T, E) -> masked embedding gradient (B, T, E), no mask grad."""
        (mask,) = ctx.saved_tensors
        return g * mask, None


class _RoutedScale(torch.autograd.Function):
    """RMSNorm's learned scale with masked parameter gradients, normal input gradients."""

    @staticmethod
    def forward(ctx, normalized: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor):
        """normalized: (B, T, ..., D), weight: (D,), broadcast mask -> same shape."""
        ctx.save_for_backward(normalized, weight, mask)
        return normalized * weight

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        """g: (B, T, ..., D) -> input grad of same shape, weight grad (D,), None."""
        normalized, weight, mask = ctx.saved_tensors
        # Freezing a norm gain must not block gradients to earlier adapters.
        gn = g * weight if ctx.needs_input_grad[0] else None
        gw = (g * normalized * mask).reshape(-1, weight.numel()).sum(0) if ctx.needs_input_grad[1] else None
        return gn, gw, None


def routed_rms_norm(x: torch.Tensor, norm: nn.Module, core_mask: torch.Tensor) -> torch.Tensor:
    """x: (B, T, ..., D), core_mask: (B, T) -> Qwen3 RMSNorm of same shape.

    Match HF's fp32 normalization exactly; only the learned gain's gradient
    contributions are masked. The extra axis in QK norms represents heads.
    """
    normalized = x.float()
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = (normalized * torch.rsqrt(variance + norm.variance_epsilon)).to(x.dtype)
    mask = core_mask.reshape(*core_mask.shape, *([1] * (x.ndim - core_mask.ndim)))
    return _RoutedScale.apply(normalized, norm.weight, mask)


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


def unit_slice(widths: list[int], k: int) -> slice:
    """Contiguous slice owned by expert ``k`` when units are laid out as ``widths`` in order."""
    start = sum(widths[:k])
    return slice(start, start + widths[k])


def unit_owner(widths: list[int], first_expert: int = 0) -> torch.Tensor:
    """Expert index of every unit, (sum(widths),), for units laid out as ``widths`` in order.

    Unit ``u`` in slice ``k`` maps to ``first_expert + k``. Integer tensor, so
    ``.to(dtype)`` on the owning module leaves it alone when registered as a buffer.
    """
    owner = torch.repeat_interleave(torch.arange(len(widths)), torch.tensor(widths, dtype=torch.long))
    return owner + first_expert


# --------------------------------------------------------------------------- #
# model                                                                       #
# --------------------------------------------------------------------------- #


class RoutedQwen3Attention(Qwen3Attention):
    """HF Qwen3 attention with expert-0 parameter gradients routed per position."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        core_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """hidden_states: (B, T, E), core_mask: (B, T) -> output (B, T, E).

        position_embeddings: cos/sin (B or 1, T, head_dim); attention_mask is
        HF's causal mask. Q/K gains apply to (B, T, heads, head_dim) tensors.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        mask = core_mask.unsqueeze(-1) # (B, T, 1), shared across projection units
        query = routed_linear(hidden_states, self.q_proj.weight, self.q_proj.bias, out_mask=mask).view(hidden_shape)
        key = routed_linear(hidden_states, self.k_proj.weight, self.k_proj.bias, out_mask=mask).view(hidden_shape)
        value = routed_linear(hidden_states, self.v_proj.weight, self.v_proj.bias, out_mask=mask).view(hidden_shape).transpose(1, 2)
        query = routed_rms_norm(query, self.q_norm, core_mask).transpose(1, 2)
        key = routed_rms_norm(key, self.k_norm, core_mask).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward,
        )
        output, weights = attention_interface(
            self, query, key, value, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
        )
        output = output.reshape(*input_shape, -1).contiguous()
        return routed_linear(output, self.o_proj.weight, self.o_proj.bias, out_mask=mask), weights


class RoutedQwen3DecoderLayer(Qwen3DecoderLayer):
    """``Qwen3DecoderLayer`` whose MLP takes the routing masks.

    Installed by swapping ``__class__`` on the stock layers (so ``deepcopy``,
    ``state_dict`` and ``torch.compile`` see an ordinary module). ``mlp`` must
    implement ``forward(x, fwd_mask, bck_mask)`` with ``(B, T, E)`` input and
    ``(B, T, K)`` masks, and ``expert_params(k)``. The body is the stock
    forward with expert masks sent to the MLP, and expert 0's backward mask
    sent to the attention projections and norm gains.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        fwd_mask: Optional[torch.Tensor] = None,
        bck_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        hidden_states:  (B, T, E)
        attention_mask: HF-prepared causal mask (or None for the SDPA is_causal path)
        fwd_mask:       (B, T, K)
        bck_mask:       (B, T, K)
        returns:        (B, T, E)
        """
        residual = hidden_states
        core_mask = bck_mask[..., 0] # (B, T), trunk and core MLP share this owner
        hidden_states = routed_rms_norm(hidden_states, self.input_layernorm, core_mask)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            core_mask=core_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = routed_rms_norm(hidden_states, self.post_attention_layernorm, core_mask)
        hidden_states = self.mlp(hidden_states, fwd_mask=fwd_mask, bck_mask=bck_mask)
        return residual + hidden_states


class RoutedTransformer(QwenTransformer):
    """Qwen3 trunk with a routed feedforward in every layer.

    Subclasses compute their adapter geometry from ``config`` and pass a
    ``make_mlp`` factory producing one feedforward module per layer (see
    ``RoutedQwen3DecoderLayer`` for the interface). ``labels[0]`` must be
    ``"core"``.
    """

    def __init__(
        self,
        config: RoutedModelConfig,
        labels: list[str],
        make_mlp: Callable[[], nn.Module],
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:

        super().__init__(config, dtype=dtype)
        assert labels[0] == "core", "first label must be core"
        self.labels = labels

        for layer in self.hf.model.layers:
            assert type(layer) is Qwen3DecoderLayer, f"unexpected layer type {type(layer).__name__}"
            layer.__class__ = RoutedQwen3DecoderLayer
            layer.self_attn.__class__ = RoutedQwen3Attention
            layer.mlp = make_mlp()
            layer.mlp.apply(self._init_adapter)

    @staticmethod
    def _init_adapter(module: nn.Module) -> None:
        """HF init (``normal(0, 0.02)``) for the adapter's linears; adapters own their ``nn.Parameter`` init."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

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

        if label == "core":
            yield from self.hf.model.embed_tokens.parameters()
            yield from self.hf.model.norm.parameters()
            if not self.config.tie_embeddings:
                yield from self.hf.lm_head.parameters()

        for layer in self.hf.model.layers:

            if label == "core":
                yield from layer.self_attn.parameters()
                yield from layer.input_layernorm.parameters()
                yield from layer.post_attention_layernorm.parameters()

            yield from layer.mlp.expert_params(k)

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
        dtype, device = self.hf.model.embed_tokens.weight.dtype, tokens.device
        fwd_mask = expand_mask(fwd_mask, B, T, K, dtype, device) # (B, T, K)
        bck_mask = expand_mask(bck_mask, B, T, K, dtype, device) # (B, T, K)

        core_mask = bck_mask[..., 0] # (B, T), no separate trunk control
        trunk = self.hf.model
        # Embedding lookup has no differentiable input to preserve: mask its
        # output gradient to route embedding-weight contributions by position.
        hidden = _ScaleGradient.apply(trunk.embed_tokens(tokens), core_mask.unsqueeze(-1))
        position_ids = torch.arange(T, device=tokens.device).unsqueeze(0)
        attention_mask = create_causal_mask(
            config=trunk.config, inputs_embeds=hidden, attention_mask=None,
            past_key_values=None, position_ids=position_ids,
        )
        position_embeddings = trunk.rotary_emb(hidden, position_ids)
        # Same no-cache Qwen3 execution order; every layer runs once on all B
        # sequences. Attention kernels and RoPE remain HF implementations.
        for layer in trunk.layers:
            hidden = layer(
                hidden, attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                fwd_mask=fwd_mask, bck_mask=bck_mask,
            )
        hidden = routed_rms_norm(hidden, trunk.norm, core_mask)
        # This also routes the output-head contribution to tied embeddings.
        logits = routed_linear(hidden, self.hf.lm_head.weight, out_mask=core_mask.unsqueeze(-1))
        loss = self.loss_fn(logits, targets) if targets is not None else None
        return logits, loss
