"""
Pure utility functions with no model or heavy framework dependencies.

This module can be safely imported by any module in the project (including
model/config.py and dataloader.py) without circular imports.

Model-dependent helpers (make_model, copy_model, etc.) live in model_utils.py.
"""

from __future__ import annotations

import json
import random
import pickle
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from src.run.util.distributed import is_main_process


def json_safe(obj):
    """Recursively convert dataclasses/Paths to JSON-safe types, skipping non-serializable fields."""
    if is_dataclass(obj) and not isinstance(obj, type):
        from dataclasses import fields
        return {f.name: json_safe(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                out[k] = json_safe(v)
            except (TypeError, NotImplementedError):
                out[k] = repr(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)


def labels_to_str(labels: Iterable[str]) -> str:
    """
    Sort labels by appending 'core' first, then sorting the rest alphabetically.
    'core' always comes first, remaining labels sorted alphabetically.
    E.g. {"core", "biology"} → "core_biology", {"core"} → "core".
    """
    labels = set(labels)
    parts = []
    if "core" in labels:
        parts.append("core")
        labels.discard("core")
    parts.extend(sorted(labels))
    return "_".join(parts)


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_line(msg: dict, log_fp: Path) -> None:
    if is_main_process():
        with open(log_fp, "a") as f:
            f.write(json.dumps(msg, default=str) + "\n")


def save_losses(res_dir: Path, losses: dict) -> None:
    if is_main_process():
        losses_np = {
            "train": np.array(losses["train"]),
            "val": {label: np.array(vals) for label, vals in losses["val"].items()},
        }
        (res_dir / "losses.pkl").write_bytes(pickle.dumps(losses_np))


def split_batch(batch: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a ``(B, T+1)`` token batch to ``device`` as inputs and next-token targets."""
    batch = batch.to(device, non_blocking=True)
    return batch[:, :-1], batch[:, 1:]


def get_exp_mask(
    labels: list[str],
    selected_labels: Optional[Iterable[str]],
    device: torch.device,
) -> torch.Tensor:
    """Create a boolean expert selection mask of length len(labels)."""
    K = len(labels)
    mask = torch.zeros(K, device=device, dtype=torch.bool)

    if selected_labels is None:
        mask[:] = True
    else:
        label_set = set(labels)
        for e in selected_labels:
            assert e in label_set, f"Unknown expert label '{e}' not in {labels}"
            mask[labels.index(e)] = True

    return mask

