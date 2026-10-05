"""
Experiment templates and model sizing utilities.

Two experiment templates are provided as factory functions that return
pre-configured ``ExperimentConfig`` instances (data + run settings). Models are
declared per stage; ``stories_model`` / ``realistic_model`` build model configs
matching each template's tokenizer and context length.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.run.util.config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
)
from src.model.utils import calc_base_params


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
    """Dense SimpleStories model (vocab 4096, ctx 512, EOS id 1) with a 4x MLP."""

    return ModelConfig(
        embed_dim=embed_dim,
        num_layers=num_layers,
        mlp_dim=4 * embed_dim,
        num_heads=num_heads,
        num_key_value=2,
        ctx_len=512,
        vocab_size=4096,
        attn_bias=True,
        eos_token_id=1,
    )


def chinchilla_tokens(model: ModelConfig, tokens_per_param: int = 20) -> int:
    """Compute-optimal token budget (Hoffmann et al., 2022): ~20 tokens per parameter,
    counting all parameters including embeddings."""

    assert model.mlp_dim == 4 * model.embed_dim, "calc_base_params assumes a 4x MLP"
    num_params = calc_base_params(model.embed_dim, model.num_layers, model.vocab_size, model.num_heads, model.num_key_value)
    return tokens_per_param * num_params
