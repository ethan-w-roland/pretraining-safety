"""25M SimpleStories comparison with one auxiliary topic.

uv run torchrun --nproc_per_node=2 -m src.run.experiment.routed.run
"""

from pathlib import Path

from src.model.config import RoutedModelConfig, model_config
from src.run.experiment.config import GetStoriesConfig, trinity_train_config
from src.run.main import run
from src.run.train.base import BaselineConfig, FilteringConfig
from src.run.train.routing import RoutingConfig
from src.run.util.config import EvalConfig, ExperimentConfig, TrainConfig


def make_config(seed: int = 1, tokens_per_param: int = 20) -> ExperimentConfig:
    """Baseline, retained-data retraining, and GR-MLP at the same token budget."""
    config = GetStoriesConfig(num_aux=1)
    base = model_config("25M")

    def train_config() -> TrainConfig:
        return trinity_train_config(base, seed=seed, tokens_per_param=tokens_per_param)

    config.res_root = Path(__file__).resolve().parents[4] / "results" / "routed" / f"seed_{seed}"
    config.compile = True
    config.stages = [
        BaselineConfig(
            name="base_25m",
            model=model_config("25M"),
            train=train_config(),
            eval=EvalConfig(elicit=False, num_train_evals=200),
        ),
        FilteringConfig(
            name="filter_25m",
            model=model_config("25M"),
            train=train_config(),
            eval=EvalConfig(elicit=True, num_train_evals=200),
            retain_targets=[["core"]],
        ),
        RoutingConfig(
            name="gr_mlp_25m",
            model=RoutedModelConfig.from_base(
                base, arch="moe", core_param_prc=0.95, aux_param_prc=0.05,
            ),
            train=train_config(),
            eval=EvalConfig(elicit=True, num_train_evals=200),
        ),
    ]
    return config


if __name__ == "__main__":
    run(make_config())
