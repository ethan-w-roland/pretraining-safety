"""
Dense transformer: Qwen3 (Qwen Team, 2025), instantiated from
``transformers.Qwen3ForCausalLM`` so the trunk is literally the reference
implementation:

- pre-norm blocks with RMSNorm (``eps=1e-6``), residual ``h = x + attn(norm(x)); out = h + mlp(norm(h))``
- grouped-query attention with QK-norm (RMSNorm over ``head_dim`` on q and k),
  no biases, ``head_dim`` decoupled from ``embed_dim / num_heads``, rotary
  embeddings with ``theta=1e6``
- SwiGLU feedforward with no biases, hidden width ``~8/3 d`` (``utils.swiglu_dim``)
- tied token embedding / output projection (as Qwen3-0.6B..4B)
- plain causal attention over the packed sequence (HF's default mask; EOS is
  just a token, as in Qwen pretraining where documents are packed with
  ``<|endoftext|>`` between them)

``BaseTransformer`` wraps the HF model behind the project's
``forward(tokens, targets) -> (logits, loss)`` interface and computes
cross-entropy on fp32 logits against the loader's already-shifted targets
(``-1000`` ignored).

Models are built directly in their training dtype (``dtype``, default bf16)
rather than cast afterwards: ``model.to(bfloat16)`` would also cast the rotary
``inv_freq`` buffer, which HF creates in fp32 on purpose, while construction
through HF's ``from_config(..., dtype=dtype)`` leaves it fp32.

The routed models (``src.model.routed``) build on the same wrapper and swap
each layer's MLP for a routed one.
"""

from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Qwen3Config

from src.model.config import ModelConfig, Transformer


def qwen3_config(config: ModelConfig) -> Qwen3Config:
    """Map the project ``ModelConfig`` onto a ``Qwen3Config`` (default RoPE, no sliding window, no KV cache)."""
    head_dim = config.head_dim if config.head_dim is not None else config.embed_dim // config.num_heads
    return Qwen3Config(
        vocab_size=config.vocab_size,
        hidden_size=config.embed_dim,
        intermediate_size=config.mlp_dim,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        num_key_value_heads=config.num_key_value,
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=config.ctx_len,
        initializer_range=0.02,
        rms_norm_eps=config.norm_eps,
        use_cache=False,
        tie_word_embeddings=config.tie_embeddings,
        rope_theta=config.rope_theta,
        attention_bias=config.attn_bias,
        use_sliding_window=False,
        eos_token_id=config.eos_token_id if config.eos_token_id >= 0 else None,
    )


class QwenTransformer(Transformer):
    """Project wrapper around a freshly initialised ``Qwen3ForCausalLM`` (``self.hf``), built in ``dtype``."""

    def __init__(self, config: ModelConfig, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(config)
        self.dtype = dtype
        self.hf = AutoModelForCausalLM.from_config(qwen3_config(config), dtype=dtype)

    @staticmethod
    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, T, V) any float dtype
        targets: (B, T) next-token ids, -1000 = ignored
        returns: scalar mean CE computed on fp32 logits
        """
        return F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1000,
            reduction="mean",
        )


class BaseTransformer(QwenTransformer):

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        tokens:  (B, T) token ids
        targets: (B, T) next-token ids, -1000 = ignored; None for no loss
        returns: (logits (B, T, V), loss scalar or None)
        """
        logits = self.hf(input_ids=tokens).logits # (B, T, V)
        loss = self.loss_fn(logits, targets) if targets is not None else None
        return logits, loss
