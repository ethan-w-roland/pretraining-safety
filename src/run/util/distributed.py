"""
Minimal DDP helpers for torchrun.

Launched with torchrun (RANK / WORLD_SIZE / LOCAL_RANK set), ``setup_distributed``
initialises an NCCL process group; launched with plain ``python``, everything
here degrades to single-process no-ops.
"""

from __future__ import annotations

import atexit
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed() -> None:
    """Initialise the process group under torchrun. Idempotent."""
    if "RANK" not in os.environ or dist.is_initialized():
        return
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    # destroy once at exit, so several run() calls in one process share the group
    atexit.register(cleanup_distributed)


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0)) #within node rank, which is different than rank in case of multi-node


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def broadcast_object(obj: object, src: int = 0) -> object:
    """Return ``src`` value of a picklable object on every rank."""
    if not is_distributed():
        return obj
    obj_list = [obj if get_rank() == src else None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """alias for broadcast_object that noop on non-distributed runs"""
    if is_distributed():
        dist.broadcast(tensor, src=src)
    return tensor


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """All-reduce ``tensor`` in place (sum, or mean if ``average``)."""
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= get_world_size()
    return tensor


def get_raw_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP and torch.compile (wrapping order: DDP(compiled(model)))."""
    if isinstance(model, DDP):
        model = model.module
    return getattr(model, "_orig_mod", model)
