"""
Model-dependent utilities: model construction, copying, saving/loading,
parameter logging, and model sizing.

Separated from run/utils.py so that run/utils.py stays dependency-free and
can be imported by low-level modules (model/config.py, dataloader.py, etc.)
without circular imports.

Saved models live at ``<res_dir>/checkpoint.pth`` and hold the weights plus
the model config, so ``load_model`` needs only the directory (and the run
config, for device placement). The directory must be on storage visible to
every rank.
"""

from __future__ import annotations

import dataclasses
import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional
import torch

from src.model.base import BaseTransformer
from src.model.config import ModelConfig, Transformer
from src.model.routed import RoutedTransformer
from src.run.util.distributed import (
    get_local_rank,
    get_raw_model,
    barrier,
    is_distributed,
    is_main_process,
    DDP,
)

if TYPE_CHECKING:
    from src.run.util.config import ExperimentConfig


def prepare_model(
    model: Transformer,
    run_config: ExperimentConfig,
) -> Transformer:
    """Move ``model`` to the run device, then compile and DDP-wrap as configured.

    Models are constructed in their training dtype (``QwenTransformer(dtype=...)``),
    so no dtype cast happens here.
    """

    model = model.to(run_config.device)

    if run_config.compile:
        run_config.logger.info(f"Compiling {type(model).__name__} with torch.compile...")
        model = torch.compile(model, dynamic=True)

    if is_distributed():
        barrier()
        local_rank = get_local_rank()
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=run_config.find_unused_parameters,
            gradient_as_bucket_view=True,
            bucket_cap_mb=100,
        )

    return model


def make_model(
    model_class: Transformer,
    model_config: ModelConfig,
    run_config: ExperimentConfig,
    extra_args: Optional[dict] = None,
) -> Transformer:
    """Build a freshly initialised ``model_class`` ready for training."""

    if extra_args is None:
        extra_args = dict()

    model = model_class(model_config, **extra_args)
    log_model_params(model, run_config.logger)
    return prepare_model(model, run_config)


def copy_model(
    model: Transformer,
    run_config: ExperimentConfig,
) -> Transformer:
    """Copy a model for finetuning or evaluation, leaving ``model`` unchanged."""

    copied_model = deepcopy(get_raw_model(model))
    return prepare_model(copied_model, run_config)


def save_model(res_dir: Path, model: Transformer) -> None:
    """Write the weights and model config to ``res_dir/checkpoint.pth``"""

    if is_main_process():
        raw_model = get_raw_model(model)
        model_config = dataclasses.asdict(raw_model.config)
        res_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": raw_model.state_dict(), "config": model_config}, str(res_dir / "checkpoint.pth"))
    barrier()


def load_model(
    res_dir: Path,
    run_config: ExperimentConfig,
    frozen: bool = False,
) -> Transformer:
    """Rebuild the dense model saved in ``res_dir/checkpoint.pth`` on the run device.

    ``frozen`` models are in eval mode without gradients and are not DDP-wrapped
    (there are no gradients to synchronize); otherwise the model is prepared
    for training like ``make_model``.
    """

    path = res_dir / "checkpoint.pth"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)

    assert checkpoint["config"]["arch"] == "base", f"load_model supports dense models only, got {checkpoint['config']['arch']}"
    model = BaseTransformer(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])

    if not frozen:
        return prepare_model(model, run_config)

    model = model.to(run_config.device).eval().requires_grad_(False)
    if run_config.compile:
        model = torch.compile(model, dynamic=True)
    return model


def expert_kwargs(model: Transformer, expert_labels: Optional[Iterable[str]]) -> dict:
    """Forward kwargs that run a routed model with only ``expert_labels`` active
    (all experts if None); ``{}`` for a dense model.

    The same mask is used forward and backward, at every position.
    """

    raw_model = get_raw_model(model)
    if not isinstance(raw_model, RoutedTransformer):
        return {}

    labels = raw_model.labels
    mask = torch.ones(len(labels), dtype=torch.bool)
    if expert_labels is not None:
        mask[:] = False
        for label in expert_labels:
            assert label in labels, f"Unknown expert label {label!r} not in {labels}"
            mask[labels.index(label)] = True
    mask = mask.to(next(raw_model.parameters()).device)
    return {"fwd_mask": mask, "bck_mask": mask}


def log_model_params(
    model: Transformer,
    logger: logging.Logger,
) -> None:
    """
    Log the number of parameters in the model.
    Logs label-specific parameter counts for routed models.
    """

    model = get_raw_model(model)
    model_type = type(model).__name__
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Initialized {model_type} with {num_params:,} total parameters")

    if isinstance(model, RoutedTransformer):
        for label in model.labels:
            num_params = sum(p.numel() for p in model.get_params(label))
            logger.info(f"  Label '{label}' has {num_params:,} parameters")


def swiglu_dim(embed_dim: int, multiple_of: int = 256) -> int:
    """LLaMA SwiGLU hidden width: ``int(2 * 4 * embed_dim / 3)`` rounded *up* to ``multiple_of``.

    Three ``embed_dim x mlp_dim`` matrices at ~``8/3 * embed_dim`` have the same
    parameter count as the two matrices of a 4x MLP (e.g. 4096 -> 11008).
    """
    hidden = int(2 * 4 * embed_dim / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


def calc_base_params(
    embed_dim: int,
    num_layers: int,
    vocab_size: int = 50304,
    num_heads: int = 8,
    num_kv: int = 2,
    mlp_dim: int | None = None,
    attn_bias: bool = False,
    head_dim: int | None = None,
    tie_embeddings: bool = True,
) -> int:
    """Exact parameter count of ``BaseTransformer`` for a configuration (Qwen3 layout).

    ``mlp_dim`` is the SwiGLU hidden width; defaults to ``swiglu_dim(embed_dim)``.
    ``head_dim`` defaults to ``embed_dim // num_heads``. Counts the QK-norm gains
    and a single embedding matrix when tied (as ``model.parameters()`` does).
    """

    d = embed_dim
    L = num_layers
    V = vocab_size
    h = swiglu_dim(d) if mlp_dim is None else mlp_dim
    hd = d // num_heads if head_dim is None else head_dim
    q_dim = num_heads * hd
    kv_dim = num_kv * hd

    attn_q = d * q_dim + (q_dim if attn_bias else 0)
    attn_k = d * kv_dim + (kv_dim if attn_bias else 0)
    attn_v = d * kv_dim + (kv_dim if attn_bias else 0)
    attn_o = q_dim * d + (d if attn_bias else 0)
    qk_norm = hd + hd
    mlp_fc_1 = d * h # no biases
    mlp_fc_2 = d * h
    mlp_proj = h * d
    norms = d + d

    layer_params = attn_q + attn_k + attn_v + attn_o + qk_norm + mlp_fc_1 + mlp_fc_2 + mlp_proj + norms

    embed = V * d
    unembed = 0 if tie_embeddings else V * d # no bias
    final_norm = d

    return embed + unembed + final_norm + L * layer_params
