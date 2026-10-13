import dataclasses
from dataclasses import dataclass
from transformers import AutoTokenizer
from torch import nn
from typing import Literal
from src.run.util.tools import json_safe

@dataclass
class ModelConfig:
    """Qwen3 dense architecture (``src.model.base`` maps this onto ``transformers.Qwen3Config``)."""

    arch: str = "base"
    tokenizer: AutoTokenizer | None = None # set at run setup from the dataset metadata
    tokenizer_name: str | None = None # HF tokenizer the config is sized for (checked against the dataset's)
    ctx_len: int = 1024
    vocab_size: int = 50304
    num_layers: int = 8
    num_heads: int = 8
    num_key_value: int = 2
    head_dim: int | None = None # Qwen3 decouples this from embed_dim / num_heads (e.g. 128 at 1024 wide); None = embed_dim // num_heads
    attn_bias: bool = False # Qwen3: no attention biases
    eos_token_id: int = -1
    embed_dim: int = 512
    mlp_dim: int = 1536 # SwiGLU hidden width: int(8/3 * embed_dim) rounded up to 256 (utils.swiglu_dim)
    norm_eps: float = 1e-6 # RMSNorm eps (Qwen3)
    rope_theta: float = 1_000_000.0 # rotary base (Qwen3)
    tie_embeddings: bool = True # Qwen3 ties embed/unembed for the small models (0.6B-4B)


# --------------------------------------------------------------------------- #
# supported sizes                                                             #
# --------------------------------------------------------------------------- #
# Two families, one architecture (``src.model.base``: Qwen3 dense).
#
# Qwen3 sizes (``"0.6B"`` .. ``"32B"``) are the released Qwen3 dense shapes
# verbatim (Qwen3 Technical Report, Table 1, and the HF ``config.json`` files),
# with the Qwen tokenizer (vocab 151,936). ``ctx_len`` is set to Qwen3's
# first-stage pretraining length (4,096), not the 40,960 inference limit.
# ``eos_token_id`` is ``<|endoftext|>`` (151643), the token Qwen packs between
# documents.
#
# SimpleStories sizes (``"25M"``, ``"50M"``, ``"100M"``) are our own shapes for
# the SimpleStories tokenizer (vocab 4,096, ctx 512, EOS 1). They follow the
# rules Qwen3 applies at its small end, read off 0.6B and 1.7B:
#
# - ``head_dim = 128`` at every size (universal in Qwen3).
# - GQA ratio 2 (0.6B and 1.7B are 16 / 8 Q / KV heads).
# - Q width ``2 * embed_dim``, i.e. ``num_heads = embed_dim / 64`` (0.6B: 16 heads
#   at 1024 wide; keeps >= 4 heads at 256 wide).
# - ``mlp_dim = 3 * embed_dim`` (0.6B, 1.7B, 8B).
# - tied embeddings (all Qwen3 <= 4B).
# - Depth roughly constant (22-25) while width grows 256 -> 384 -> 512, as Qwen
#   holds 28 layers from 0.6B to 1.7B and scales width; the family then
#   continues to 28 layers at 1024 wide. This extrapolates Qwen's trend of
#   getting relatively deeper as models get smaller (``embed_dim / num_layers``
#   is 128 at 14B, 37 at 0.6B) rather than freezing the aspect ratio.
#
# Exact counts (``utils.calc_base_params``, verified against the HF models):
#
#     size   embed  layers  heads   mlp    total      non-embedding
#     25M      256      24   4/2    768    24.66M     23.61M
#     50M      384      22   6/3   1152    50.26M     48.68M
#     100M     512      25   8/4   1536   100.43M     98.34M
#     0.6B    1024      28  16/8   3072    0.596B     0.440B
#     1.7B    2048      28  16/8   6144    1.721B     1.409B
#     4B      2560      36  32/8   9728    4.022B     3.634B
#     8B      4096      36  32/8  12288    8.191B     6.946B
#     14B     5120      40  40/8  17408   14.768B    13.212B
#     32B     5120      64  64/8  25600   32.762B    31.206B

STORIES_TOKENIZER = "SimpleStories/SimpleStories-1.25M"
QWEN3_TOKENIZER = "Qwen/Qwen3-0.6B" # shared by every Qwen3 size


def _stories(embed_dim: int, num_layers: int) -> ModelConfig:
    """SimpleStories config under the Qwen3 small-model rules (see the section comment above)."""
    assert embed_dim % 128 == 0, "embed_dim must be a multiple of 128 for head_dim 128 with an even head count"
    num_heads = embed_dim // 64
    return ModelConfig(
        tokenizer_name=STORIES_TOKENIZER,
        vocab_size=4096,
        ctx_len=512,
        eos_token_id=1,
        embed_dim=embed_dim,
        num_layers=num_layers,
        mlp_dim=3 * embed_dim,
        num_heads=num_heads,
        num_key_value=num_heads // 2,
        head_dim=128,
        tie_embeddings=True,
    )


def _qwen3(embed_dim: int, num_layers: int, mlp_dim: int, num_heads: int, tie_embeddings: bool) -> ModelConfig:
    """A released Qwen3 dense shape."""
    return ModelConfig(
        tokenizer_name=QWEN3_TOKENIZER,
        vocab_size=151936,
        ctx_len=4096,
        eos_token_id=151643,
        embed_dim=embed_dim,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_key_value=8,
        head_dim=128,
        tie_embeddings=tie_embeddings,
    )


MODEL_SIZES: dict[str, ModelConfig] = {
    # SimpleStories
    "25M": _stories(256, 24),
    "50M": _stories(384, 22),
    "100M": _stories(512, 25),
    # Qwen3 dense
    "0.6B": _qwen3(1024, 28, 3072, 16, tie_embeddings=True),
    "1.7B": _qwen3(2048, 28, 6144, 16, tie_embeddings=True),
    "4B": _qwen3(2560, 36, 9728, 32, tie_embeddings=True),
    "8B": _qwen3(4096, 36, 12288, 32, tie_embeddings=False),
    "14B": _qwen3(5120, 40, 17408, 40, tie_embeddings=False),
    "32B": _qwen3(5120, 64, 25600, 64, tie_embeddings=False),
}

STORIES_SIZES = ("25M", "50M", "100M")
QWEN3_SIZES = ("0.6B", "1.7B", "4B", "8B", "14B", "32B")


def model_config(size: str, **overrides) -> ModelConfig:
    """A fresh ``ModelConfig`` for a supported size, with optional field overrides."""
    if size not in MODEL_SIZES:
        raise KeyError(f"unknown model size {size!r}; supported: {list(MODEL_SIZES)}")
    return dataclasses.replace(MODEL_SIZES[size], **overrides)


ROUTED_ARCHS = ("moe", "lora")


@dataclass
class RoutedModelConfig(ModelConfig):
    """Configuration for the gradient-routed models (``src.model.moe``, ``src.model.lora``).

    ``core_dim = core_param_prc * mlp_dim`` and ``aux_dim = aux_param_prc * mlp_dim``
    define the core and combined auxiliary MLP budgets. Auxiliary budget is
    split among labels. Core width rounds to 64; when shares sum to one,
    auxiliary width is the exact remainder, preserving the dense MLP width.

    - ``moe``: one SwiGLU of hidden width ``core_dim + aux_dim``, partitioned
      among labels. Total parameters exactly match dense when shares sum to 1.
    - ``lora``: a core SwiGLU of width ``core_dim`` plus one LoRA per linear
      whose rank units are partitioned among aux labels; ``lora_rank`` (per aux
      expert) defaults to the rank approximating each aux label's share of the
      combined auxiliary parameter budget; integer ranks prevent exact matching.
    """
    arch: Literal["moe", "lora"] = "moe"
    core_param_prc: float = 0.95
    aux_param_prc: float = 0.05
    lora_rank: int | None = None

    @classmethod
    def from_base(cls, base: ModelConfig, **overrides) -> "RoutedModelConfig":
        """Create a RoutedModelConfig inheriting all values from a ModelConfig."""
        base_fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
        base_fields.update(overrides)
        return cls(**base_fields)

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
