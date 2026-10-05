"""
uv run torchrun --nproc_per_node=2 -m src.run.experiment.distill.run
"""

import json
import os
from pathlib import Path
import torch

from src.run.train.base import BaselineConfig, FilteringConfig
from src.run.train.distill import DistillConfig
from src.run.experiment.config import GetStoriesConfig, chinchilla_tokens, stories_model
from src.run.util.config import EvalConfig, ExperimentConfig, TrainConfig
from src.run.main import run

torch.cuda.empty_cache()

root_dir = Path("src").absolute()

LR = 1e-4
BS = 128 # effective batch size in seqs, split over GPUs
NUM_GPUS = int(os.environ.get("WORLD_SIZE", 1))

M20 = stories_model(embed_dim=448, num_layers=9) # 22.7M params, w2d 49.8
M5 = stories_model(embed_dim=256, num_layers=5) # 5.6M params, w2d 51.2

no_elicit = EvalConfig(elicit=False, num_train_evals=200)
elicit = EvalConfig(elicit=True, num_train_evals=200)

def make_config(seed: int, tokens_per_param: int = 20) -> ExperimentConfig:

    config = GetStoriesConfig(num_aux=1)
    FORGET = config.data.aux[0]
    RETAIN = [["core"]] # "core" is the merged label of all core topics
    metadata = json.load(open(config.data.dirs[0] / "metadata.json"))

    # pretraining stages get tokens_per_param tokens per parameter; 20 is Chinchilla-optimal
    # (~454M tokens for 20M, ~111M for 5M)
    train_20m = TrainConfig(lr=LR, bs=BS // NUM_GPUS, seed=seed, num_tokens=chinchilla_tokens(M20, tokens_per_param))
    train_5m = TrainConfig(lr=LR, bs=BS // NUM_GPUS, seed=seed, num_tokens=chinchilla_tokens(M5, tokens_per_param))

    # the 20M baseline samples uniformly over all train data, so it sees the forget topic's
    # share of its budget (~9.6M tokens at 20/param); the specialist fine-tunes on that many forget tokens
    forget_share = metadata[FORGET]["train"]["total_tokens"] / metadata["all"]["total_tokens_train"]
    specialist = TrainConfig(
        lr=LR / 4,
        bs=BS // NUM_GPUS,
        seed=seed,
        num_tokens=round(train_20m.num_tokens * forget_share),
    )

    prefix = f"distill/seed_{seed}"
    config.res_root = root_dir.parent / "results" / prefix

    config.log_level = "DEBUG"
    config.compile = True
    config.stages = [
        #20M params
        BaselineConfig(
            name="base_20m",
            model=M20,
            train=train_20m,
            eval=no_elicit,
        ),
        FilteringConfig(
            name="filter_20m",
            model=M20,
            train=train_20m,
            eval=elicit,
            retain_targets=RETAIN,
        ),
        #5M params
        BaselineConfig(
            name="base_5m",
            model=M5,
            train=train_5m,
            eval=no_elicit,
        ),
        FilteringConfig(
            name="filter_5m",
            model=M5,
            train=train_5m,
            eval=elicit,
            retain_targets=RETAIN,
        ),
        DistillConfig(
            name="distill_all_5m",
            model=M5,
            train=train_5m,
            eval=elicit,
            retain_targets=[["core", FORGET]],
            teacher="base_20m",
            teacher_type="baseline",
        ),
        DistillConfig(
            name="distill_base_5m",
            model=M5,
            train=train_5m,
            eval=elicit,
            retain_targets=RETAIN,
            teacher="base_20m",
            teacher_type="baseline",
        ),
        DistillConfig(
            name="distill_filter_5m",
            model=M5,
            train=train_5m,
            eval=elicit,
            retain_targets=RETAIN,
            teacher="filter_20m",
            teacher_type="filtering",
        ),
        DistillConfig(
            name="distill_diff_5m",
            model=M5,
            train=train_5m,
            eval=elicit,
            retain_targets=RETAIN,
            teacher="base_20m",
            teacher_type="difference",
            specialist=specialist,
            alpha=1.0,
        ),
    ]

    return config


if __name__ == "__main__":
    for seed in [1]:
        run(make_config(seed))
