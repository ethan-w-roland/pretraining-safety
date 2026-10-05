"""Token data: shard files are found once at setup and cut into seqs per loader."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from src.run.util.distributed import get_rank, get_world_size

if TYPE_CHECKING:
    from src.run.util.config import ExperimentConfig, StageConfig


class Shard(Dataset):
    """A uint16 token file, cut into non-overlapping seqs of ``ctx_len + 1`` tokens.

    Each item is ``(label_idx, seq)``, where ``label_idx`` is the index of the
    shard's label in ``config.data.labels``.
    """

    def __init__(self, path: Path, ctx_len: int, label_idx: int):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.ctx_len = ctx_len
        self.label_idx = label_idx

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.ctx_len

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        start = idx * self.ctx_len
        seq = self.tokens[start : start + self.ctx_len + 1]
        return self.label_idx, torch.from_numpy(seq.astype(np.int64))


class Loader(DataLoader):
    """Shuffled ``(labels, x, y)`` batches on ``device``, with each rank getting a disjoint share.

    ``labels`` has shape ``(bs,)`` and holds each seq's index in ``config.data.labels``.
    """

    def __init__(self, dataset: Dataset, bs: int, seed: int, device: torch.device):
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            seed=seed,
        )
        super().__init__(dataset, batch_size=bs, sampler=sampler, pin_memory=True)
        self.device = device

    def reset(self, epoch: int) -> None:
        """Reshuffle with the order of ``epoch``."""
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        for labels, seqs in super().__iter__():
            labels = labels.to(self.device, non_blocking=True)
            seqs = seqs.to(self.device, non_blocking=True)
            x = seqs[:, :-1]
            y = seqs[:, 1:]
            yield labels, x, y


def make_datasets(
    data_dirs: list[Path],
    core_labels: list[str],
) -> dict[str, dict[str, list[Path]]]:
    """Find the shard files in ``data_dirs``, grouped by label and split.

    Shards are named ``<label>_<split>.bin`` or ``<label>_<split>_<idx>.bin``
    (e.g. ``fineweb_train_003.bin``); other files are ignored. An extra
    ``"core"`` label holds the shards of every label in ``core_labels``.

    Returns:
        ``{label: {"train": [paths], "test": [paths]}}``
    """
    shard_name = re.compile(r"^(?P<label>.+?)_(?P<split>train|test)(?:_\d+)?\.bin$")

    datasets: dict[str, dict[str, list[Path]]] = {}
    for data_dir in data_dirs:
        for path in sorted(Path(data_dir).glob("*.bin")):
            match = shard_name.match(path.name)
            if match is None:
                continue

            label = match["label"]
            split = match["split"]
            if label not in datasets:
                datasets[label] = {"train": [], "test": []}
            datasets[label][split].append(path)

    assert "core" not in datasets, "'core' is reserved for the core labels; rename that shard"
    for label in core_labels:
        assert label in datasets, f"core label {label!r} has no shards in {data_dirs}"

    datasets["core"] = {"train": [], "test": []}
    for label in core_labels:
        for split in ("train", "test"):
            datasets["core"][split].extend(datasets[label][split])

    return datasets


def repeat_to_length(indices: np.ndarray, length: int) -> np.ndarray:
    """Repeat ``indices`` end to end and keep the first ``length`` entries."""
    num_repeats = math.ceil(length / len(indices))
    return np.tile(indices, num_repeats)[:length]


def make_loader(
    config: ExperimentConfig,
    stage: StageConfig,
    labels: Iterable[str],
    split: str,
    balance: bool = False,
) -> Loader:
    """A loader whose batches mix seqs from all ``labels``.

    Labels contribute in proportion to their size, or equally if ``balance``
    (smaller labels are repeated). On the train split, ``stage.train.num_tokens``
    picks how many seqs to draw at random (``-1`` for all of them). Each
    epoch is a whole number of optimizer steps.
    """
    train = stage.train
    logger = config.logger
    labels = list(labels)

    parts = []
    for label in labels:
        label_idx = config.data.labels.index(label)
        shards = []
        for path in config.data.datasets[label][split]:
            shards.append(Shard(path, train.ctx_len, label_idx))
        parts.append(ConcatDataset(shards))

    if balance:
        largest = max(len(part) for part in parts)
        for i, part in enumerate(parts):
            if len(part) < largest:
                logger.warning(
                    f"Resampling {labels[i]!r} ({split}) to balance labels: "
                    f"{len(part):,} -> {largest:,} seqs ({largest / len(part):.2f}x)"
                )
            indices = repeat_to_length(np.arange(len(part)), largest)
            parts[i] = Subset(part, indices)

    dataset = ConcatDataset(parts)

    num_seqs = len(dataset)
    if split == "train" and train.num_tokens > 0:
        num_seqs = train.num_tokens // train.ctx_len

    seqs_per_step = train.bs * train.acc_steps * get_world_size()
    num_seqs -= num_seqs % seqs_per_step

    if num_seqs > len(dataset):
        logger.warning(
            f"Resampling {labels} ({split}) to fill the token budget: "
            f"{len(dataset):,} -> {num_seqs:,} seqs ({num_seqs / len(dataset):.2f}x)"
        )

    # every seq is used once before any is repeated
    order = np.random.default_rng(train.seed).permutation(len(dataset))
    indices = repeat_to_length(order, num_seqs)

    return Loader(Subset(dataset, indices), train.bs, train.seed, config.device)
