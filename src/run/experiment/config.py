"""
Experiment templates and model sizing utilities.

Two experiment templates are provided as factory functions that return
pre-configured ``ExperimentConfig`` instances (data + run settings). Models are
declared per stage; ``stories_model`` / ``realistic_model`` build model configs
matching each template's tokenizer and context length.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from src.run.util.config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    TrainConfig,
)
from src.model.utils import calc_base_params, swiglu_dim
from src.run.train.optim import adamw_betas
from src.model.config import model_config


# --------------------------------------------------------------------------- #
# Experiment templates                                                         #
# --------------------------------------------------------------------------- #

root_dir = Path("src").absolute()

def GetStoriesConfig(num_aux: int = 4) -> ExperimentConfig:
    """SimpleStories experiment: the first ``num_aux`` topics (sorted) are aux, the rest core."""

    data_dir = root_dir / "data/stories"
    metadata = json.load(open(data_dir / "metadata.json"))
    all_labels = sorted(metadata["all"]["labels"])

    return ExperimentConfig(
        data=DataConfig(
            dirs=[data_dir],
            core=all_labels[num_aux:],
            aux=all_labels[:num_aux],
        ),
    )


def stories_model(embed_dim: int, num_layers: int, num_heads: int = 8) -> ModelConfig:
    """Qwen3-style SimpleStories model (vocab 4096, ctx 512, EOS id 1)."""

    return ModelConfig(
        embed_dim=embed_dim,
        num_layers=num_layers,
        mlp_dim=swiglu_dim(embed_dim),
        num_heads=num_heads,
        num_key_value=2,
        ctx_len=512,
        vocab_size=4096,
        eos_token_id=1,
    )


def chinchilla_tokens(model: ModelConfig, tokens_per_param: int = 20) -> int:
    """Compute-optimal token budget (Hoffmann et al., 2022): ~20 tokens per parameter,
    counting all parameters including embeddings."""

    num_params = calc_base_params(
        model.embed_dim, model.num_layers, model.vocab_size, model.num_heads, model.num_key_value,
        mlp_dim=model.mlp_dim, attn_bias=model.attn_bias, head_dim=model.head_dim,
        tie_embeddings=model.tie_embeddings,
    )
    return tokens_per_param * num_params


def trinity_train_config(
    model: ModelConfig,
    seed: int = 1,
    tokens_per_param: int = 20,
    world_size: int | None = None,
) -> TrainConfig:
    """Trinity-inspired extrapolation for small Stories pretraining.

    Source: https://arxiv.org/html/2602.17004v1#S3.SS4.SSS2
    Keep Nano/Mini's base Muon LR, with PyTorch's original shape adjustment.
    AdamW scales as width**(-log2(1.5)), fitting Nano (1024, 3e-4) and Mini
    (2048, 2e-4). These small-model rates are extrapolations, not published optima.

    Trinity does not give a batch-size scaling law. We anchor the 25M Stories
    preset at 128 global sequences and scale token batch linearly with parameter
    count, rounded to a power of two. Microbatches are capped at 8 sequences per
    GPU and gradient accumulation fills the global batch. AdamW betas follow
    Porian et al. for that global token batch (``optim.adamw_betas``). WSD uses
    <=2000 warmup steps, capped at 10% of the run, and a final 10% linear decay
    to zero.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1)) if world_size is None else world_size
    assert world_size > 0
    assert tokens_per_param > 0
    params = chinchilla_tokens(model, 1)
    reference = model_config("25M")
    reference_params = chinchilla_tokens(reference, 1)
    target_sequences = 128 * params / reference_params * reference.ctx_len / model.ctx_len
    global_batch = max(1, 2 ** round(math.log2(target_sequences)))
    assert global_batch % world_size == 0, "WORLD_SIZE must divide the target global batch"
    per_rank_batch = global_batch // world_size
    micro_batch = min(8, per_rank_batch)
    assert per_rank_batch % micro_batch == 0
    adam_lr = max(2e-4, 3e-4 * (1024 / model.embed_dim) ** (math.log(1.5) / math.log(2)))
    return TrainConfig(
        adamw_lr=adam_lr,
        adamw_betas=adamw_betas(global_batch * model.ctx_len),
        muon_lr=1e-3,
        muon_momentum=0.95,
        muon_ns_steps=5,
        weight_decay=0.1,
        bs=micro_batch,
        acc_steps=per_rank_batch // micro_batch,
        ctx_len=model.ctx_len,
        warmup_prc=0.1,
        warmup_steps=2000,
        decay_prc=0.1,
        min_lr_ratio=0.0,
        seed=seed,
        num_tokens=chinchilla_tokens(model, tokens_per_param),
    )
