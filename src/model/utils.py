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
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import torch

from src.model.base import BaseTransformer
from src.model.config import ModelConfig, Transformer
from src.run.util.distributed import (
    get_local_rank,
    get_raw_model,
    barrier,
    is_main_process,
    DDP,
)

if TYPE_CHECKING:
    from src.run.util.config import ExperimentConfig


def prepare_model(
    model: Transformer,
    run_config: ExperimentConfig,
) -> Transformer:
    """Move ``model`` to the run device in bf16, then compile and DDP-wrap as configured."""

    model = model.to(run_config.device, dtype=torch.bfloat16)

    if run_config.compile:
        run_config.logger.info(f"Compiling {type(model).__name__} with torch.compile...")
        model = torch.compile(model, dynamic=True)

    if run_config.is_ddp:
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
        model_config = {
            f.name: getattr(raw_model.config, f.name)
            for f in dataclasses.fields(raw_model.config)
            if f.name != "tokenizer"
        }
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

    model = model.to(run_config.device, dtype=torch.bfloat16).eval().requires_grad_(False)
    if run_config.compile:
        model = torch.compile(model, dynamic=True)
    return model


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

    if model_type in ("MoETransformer", "LoRATransformer", "DemixTransformer"):
        label_list = list(model.labels)
        if model_type == "DemixTransformer":
            label_list = label_list + ["SHARED"]
        for label in label_list:
            num_params = sum(p.numel() for p in model.get_params(label))
            logger.info(f"  Label '{label}' has {num_params:,} parameters")


def calc_lora_rank(
    embed_dim: int,
    core_dim: int,
    aux_dim: int,
) -> int:

    moe_aux_params = 2 * embed_dim * aux_dim + aux_dim + embed_dim
    lora_params = 2 * (embed_dim + core_dim)
    lora_rank = max(1, round(moe_aux_params / lora_params))

    return lora_rank


def calc_base_params(
    embed_dim: int,
    num_layers: int,
    vocab_size: int = 50304,
    num_heads: int = 8,
    num_kv: int = 2,
) -> int:
    """For a given architectural configuration, calculate the exact parameter count."""

    d = embed_dim
    L = num_layers
    V = vocab_size
    head_dim = d // num_heads

    attn_q = d * d + d
    attn_kv = d * (2 * num_kv * head_dim) + (2 * num_kv * head_dim)
    attn_o = d * d + d
    mlp_fc = d * 4 * d + 4 * d
    mlp_proj = 4 * d * d + d
    norms = d + d

    layer_params = attn_q + attn_kv + attn_o + mlp_fc + mlp_proj + norms

    embed = V * d
    unembed = V * d + V
    final_norm = d

    return embed + unembed + final_norm + L * layer_params


def find_base_params(
    target_params: int,
    V: int = 50304,
    align: int = 32
) -> dict[str, int]:

    """Find (embed_dim, num_layers, mlp_dim) for a target parameter count.

    All returned dimensions are rounded to multiples of ``align``.

    Maintains L/d ratio within +/-10% of 0.015, which keeps the depth/width
    balance consistent across model scales.  This matters for scaling law
    experiments where shape distortion would confound the results.

    Uses Newton's method on the cubic parameter-count equation to get an
    initial estimate, then searches nearby grid points.

    Args:
        target_params: Desired total parameter count.
        V: Vocabulary size (rounded up to nearest 64 internally).
        align: All dimensions (embed_dim, mlp_dim) are rounded to multiples
            of this value.  Must be a multiple of (num_heads * 4) to ensure
            head_dim is divisible by 4 (required by rotary embeddings).
            Default 32 (compatible with num_heads=8).

    Returns:
        {"embed_dim": int, "num_layers": int, "mlp_dim": int, "vocab_size": int}
    """
    r = 0.015
    tol = 0.1
    V = 64 * ((V + 63) // 64)

    d = ((target_params - V) / (10.5 * r)) ** (1/3)
    for _ in range(10):
        f = 10.5 * r * d**3 + 2 * V * d - (target_params - V)
        d -= f / (31.5 * r * d**2 + 2 * V)

    base_d = align * round(d / align)
    candidates = []
    for embed_dim in range(max(align, base_d - 3 * align), base_d + 4 * align, align):
        L_min = max(2, math.ceil(embed_dim * r * (1 - tol)))
        L_max = math.floor(embed_dim * r * (1 + tol))
        for num_layers in range(L_min, L_max + 1):
            params = calc_base_params(embed_dim, num_layers, V)
            candidates.append((abs(params - target_params), embed_dim, num_layers))

    _, best_d, best_L = min(candidates) if candidates else (0, max(align, base_d), 2)
    mlp_dim = align * round((best_d * 4) / align)
    return {"embed_dim": best_d, "num_layers": best_L, "mlp_dim": mlp_dim, "vocab_size": V}