"""
uv run torchrun --nproc_per_node=2 -m src.run.experiment.distill-diff.run
"""

import json
from pathlib import Path
import torch

from src.run.train.base import BaselineConfig, FilteringConfig
from src.run.train.distill import DistillConfig
from src.model.config import model_config
from src.run.experiment.config import GetStoriesConfig, trinity_train_config
from src.run.util.config import EvalConfig, ExperimentConfig, TrainConfig
from src.run.main import run

torch.cuda.empty_cache()

root_dir = Path("src").absolute()

M50 = model_config("50M")
M25 = model_config("25M")

no_elicit = EvalConfig(elicit=False, num_train_evals=200)
elicit = EvalConfig(elicit=True, num_train_evals=200)

def make_config(seed: int, tokens_per_param: int = 20) -> ExperimentConfig:

    config = GetStoriesConfig(num_aux=1)
    FORGET = config.data.aux[0]
    RETAIN = [["core"]] # "core" is the merged label of all core topics
    metadata = json.load(open(config.data.dirs[0] / "metadata.json"))

    # pretraining stages get tokens_per_param tokens per parameter; 20 is Chinchilla-optimal
    # (~1.005B tokens for 50M, ~493M for 25M)
    train_50m = trinity_train_config(M50, seed=seed, tokens_per_param=tokens_per_param)
    train_25m = trinity_train_config(M25, seed=seed, tokens_per_param=tokens_per_param)

    # the 50M baseline samples uniformly over all train data; the specialist
    # fine-tunes on the forget topic's share of that baseline's token budget
    forget_share = metadata[FORGET]["train"]["total_tokens"] / metadata["all"]["total_tokens_train"]
    specialist = TrainConfig(
        # Trinity's SFT uses one tenth of the pretraining peak learning rates.
        lr=train_50m.lr / 10,
        muon_lr=train_50m.muon_lr / 10,
        bs=train_50m.bs,
        acc_steps=train_50m.acc_steps,
        seed=seed,
        num_tokens=round(train_50m.num_tokens * forget_share),
    )

    prefix = f"distill/seed_{seed}"
    config.res_root = root_dir.parent / "results" / prefix

    config.log_level = "DEBUG"
    config.compile = True
    config.stages = [
        #50M params
        BaselineConfig(
            name="base_50m",
            model=M50,
            train=train_50m,
            eval=no_elicit,
        ),
        FilteringConfig(
            name="filter_50m",
            model=M50,
            train=train_50m,
            eval=elicit,
            retain_targets=RETAIN,
        ),
        #25M params
        BaselineConfig(
            name="base_25m",
            model=M25,
            train=train_25m,
            eval=no_elicit,
        ),
        FilteringConfig(
            name="filter_25m",
            model=M25,
            train=train_25m,
            eval=elicit,
            retain_targets=RETAIN,
        ),
        DistillConfig(
            name="distill_all_25m",
            model=M25,
            train=train_25m,
            eval=elicit,
            retain_targets=[["core", FORGET]],
            teacher="base_50m",
            teacher_type="baseline",
        ),
        DistillConfig(
            name="distill_base_25m",
            model=M25,
            train=train_25m,
            eval=elicit,
            retain_targets=RETAIN,
            teacher="base_50m",
            teacher_type="baseline",
        ),
        DistillConfig(
            name="distill_filter_25m",
            model=M25,
            train=train_25m,
            eval=elicit,
            retain_targets=RETAIN,
            teacher="filter_50m",
            teacher_type="filtering",
        ),
        DistillConfig(
            name="distill_diff_25m",
            model=M25,
            train=train_25m,
            eval=elicit,
            retain_targets=RETAIN,
            teacher="base_50m",
            teacher_type="difference",
            specialist=specialist,
            alpha=1.0,
        ),
    ]

    return config


if __name__ == "__main__":
    for seed in [1]:
        run(make_config(seed))
