"""
export OMP_NUM_THREADS=16 && torchrun --nproc_per_node=8 -m src.run.main

main.py — multi-stage pretraining pipeline.

Architecture
------------
The pipeline runs a sequence of *stages*, each producing checkpoints in its
own subdirectory under the results root. Stages are typed configs
(``BaselineConfig``, ``FilteringConfig``, ``DistillConfig``, ``RoutingConfig``)
listed in ``ExperimentConfig.stages``; see ``src/run/experiment/*/run.py``.

Execution flow::

    run()                   # entry point: sets up DDP, calls setup()
      -> run_experiments()  # iterates over stages, dispatches to runners
          -> run_baseline   # train from scratch on all data
          -> run_filtering  # retrain without target data (per target group)
          -> run_distill    # KL-distill a fresh student from a frozen teacher on retained data
          -> run_routing    # train a gradient-routed model on all data

Each stage runner is self-contained: it creates the model, trains it with the
appropriate ``do_*`` function (which writes the final ``checkpoint.pth``), then
evaluates it with ``do_eval``. Every run trains all of its stages from scratch
into a fresh results directory; later stages may load earlier stages'
checkpoints.

Adversarial fine-tuning
-----------------------
Stages with ``eval.elicit=True`` run "adversarial fine-tuning" after the main
training: the model is fine-tuned on each forget-target label individually and
evaluated to measure how easily the model can re-learn the information. This
is an *elicitation* metric.
"""

import dataclasses
import gc
import json
import warnings
import torch

from src.model.base import BaseTransformer
from src.model.moe import MoETransformer
from src.model.config import Transformer
from src.model.utils import copy_model, load_model, make_model
from src.run.eval import do_eval
from src.run.util.tools import json_safe, labels_to_str, set_seeds
from src.run.util.config import ExperimentConfig, StageConfig, setup
from src.run.util.distributed import barrier, get_raw_model

from src.run.train.base import BaselineConfig, FilteringConfig, do_train
from src.run.train.elicit import do_elicit
from src.run.train.distill import DistillConfig, do_distill
from src.run.train.routing import RoutingConfig, do_routing, expert_configurations
warnings.filterwarnings("ignore", message=r"(?s).*Online softmax is disabled.*", category=UserWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def free_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def eval_log(retained: list[str]) -> dict:
    """Extra ``stats.jsonl`` fields for a model's post-training eval."""
    return {"retained": retained, "finetune": None, "elicited": False}


def get_retain_targets(stage: FilteringConfig | DistillConfig, config: ExperimentConfig) -> list[list[str]]:
    """The stage's retain targets; by default core alone, then core plus each aux label."""
    if stage.retain_targets is not None:
        return stage.retain_targets
    aux = config.data.aux
    if len(aux) == 1:
        return [["core"]]
    return [["core"]] + [["core", x] for x in aux]


def run_elicitation(
    stage: StageConfig,
    model: Transformer,
    config: ExperimentConfig,
    data_labels: list[str],
    retained: list[str],
    expert_labels: list[str] | None = None,
) -> None:
    """For each label, fine-tune a fresh copy of ``model`` on it and log the
    best loss reached (see ``do_elicit``)."""

    config.logger.info(f"{stage.name} - ADVERSARIAL FT START")

    # keep the original off the GPU while its copies train
    model = get_raw_model(model).to("cpu")
    free_memory()

    for label in sorted(data_labels):
        ft_model = copy_model(model, config)
        do_elicit(
            stage=stage,
            model=ft_model,
            config=config,
            data_labels=[label],
            expert_labels=expert_labels,
            log_args={"finetune": label, "elicited": True, "retained": retained},
        )
        del ft_model
        free_memory()


def load_teachers(
    stage: DistillConfig,
    iter_stage: DistillConfig,
    config: ExperimentConfig,
    removed: list[str],
) -> tuple[Transformer, Transformer | None]:
    """Frozen ``(teacher, anti_teacher)`` for one retain target of a distill stage.

    ``baseline``: the teacher stage's model. ``filtering``: the teacher stage's
    model for the same retain target. ``difference``: the teacher stage's model,
    with an anti-teacher fine-tuned from it on the ``removed`` labels.
    """

    teacher_dir = config.res_dir / stage.teacher

    if stage.teacher_type == "baseline":
        return load_model(teacher_dir, config, frozen=True), None

    if stage.teacher_type == "filtering":
        return load_model(teacher_dir / iter_stage.res_dir.name, config, frozen=True), None

    if stage.teacher_type == "difference":
        config.logger.info(f"Distill: fine-tuning specialist on {removed}")
        spec_stage = dataclasses.replace(
            stage,
            train=stage.specialist,
            eval=dataclasses.replace(stage.eval, num_train_evals=0),
            res_dir=iter_stage.res_dir / "specialist",
        )
        specialist = load_model(teacher_dir, config)
        do_train(spec_stage, specialist, config, train_labels=removed)
        del specialist
        free_memory()
        return load_model(teacher_dir, config, frozen=True), load_model(spec_stage.res_dir, config, frozen=True)

    raise ValueError(f"Unknown teacher_type: {stage.teacher_type}")


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_baseline(stage: BaselineConfig, config: ExperimentConfig) -> None:
    """Train from scratch on all data."""

    labels = sorted(config.data.labels)

    set_seeds(stage.train.seed)
    model = make_model(BaseTransformer, stage.model, config)
    model = do_train(stage, model, config, train_labels=labels)
    do_eval(stage, model, config, log=eval_log(labels))

    if stage.eval.elicit:
        run_elicitation(stage, model, config, data_labels=config.data.aux, retained=labels)


def run_filtering(stage: FilteringConfig, config: ExperimentConfig) -> None:
    """Data-filtering stage: retrain from scratch on only the retained labels."""

    for retained in get_retain_targets(stage, config):

        removed = sorted(set(config.data.labels) - set(retained))
        if not removed:
            continue

        config.logger.info(f"Filtering: retaining {retained}, removing {removed}")
        iter_stage = dataclasses.replace(stage, res_dir=stage.res_dir / labels_to_str(retained))

        set_seeds(stage.train.seed)
        model = make_model(BaseTransformer, stage.model, config)
        model = do_train(iter_stage, model, config, train_labels=retained)
        do_eval(iter_stage, model, config, log=eval_log(sorted(retained)))

        if stage.eval.elicit:
            run_elicitation(stage, model, config, data_labels=removed, retained=sorted(retained))

        del model
        free_memory()


def run_distill(stage: DistillConfig, config: ExperimentConfig) -> None:
    """Distillation stage: per retain target, train a fresh student on KL to a
    frozen teacher over only the retained data.
    """

    for retained in get_retain_targets(stage, config):

        removed = sorted(set(config.data.labels) - set(retained))

        config.logger.info(f"Distill: retaining {retained}, removing {removed}")
        iter_stage = dataclasses.replace(stage, res_dir=stage.res_dir / labels_to_str(retained))

        teacher, anti_teacher = load_teachers(stage, iter_stage, config, removed)

        set_seeds(stage.train.seed)
        model = make_model(BaseTransformer, stage.model, config)
        model = do_distill(iter_stage, model, config, train_labels=retained, teacher=teacher, anti_teacher=anti_teacher)
        do_eval(iter_stage, model, config, log=eval_log(sorted(retained)))

        del teacher, anti_teacher
        free_memory()

        if stage.eval.elicit and removed:
            run_elicitation(stage, model, config, data_labels=removed, retained=sorted(retained))

        del model
        free_memory()


def run_routing(stage: RoutingConfig, config: ExperimentConfig) -> None:
    """Train GR-MLP, evaluate each expert configuration, and elicit core only."""

    assert stage.model.arch == "moe", "RoutingConfig currently supports GR-MLP (moe)"
    labels = config.data.labels

    set_seeds(stage.train.seed)
    model = make_model(MoETransformer, stage.model, config, extra_args={"labels": labels})
    model = do_routing(stage, model, config)
    for experts in expert_configurations(labels):
        do_eval(stage, model, config, expert_labels=experts, log=eval_log(experts))

    if stage.eval.elicit:
        run_elicitation(
            stage, model, config,
            data_labels=config.data.aux,
            retained=["core"],
            expert_labels=["core"],
        )


# ---------------------------------------------------------------------------
# Pipeline dispatcher
# ---------------------------------------------------------------------------

RUNNERS = {
    BaselineConfig: run_baseline,
    FilteringConfig: run_filtering,
    DistillConfig: run_distill,
    RoutingConfig: run_routing,
}


def run_experiments(config: ExperimentConfig) -> None:
    """Execute the full multi-stage pipeline."""

    logger = config.logger
    stages = config.stages

    # fail before any training if a stage has no runner
    for stage in stages:
        assert type(stage) in RUNNERS, f"Stage {stage.name!r} has unknown type {type(stage).__name__}"

    for stage in stages:

        stage_dict = json_safe(stage)
        logger.info(f"Stage [{stage.name}] start: {json.dumps(stage_dict, default=str, ensure_ascii=False)}")

        runner = RUNNERS[type(stage)]
        runner(stage, config)

        free_memory()
        barrier()


def run(config: ExperimentConfig) -> None:
    """Run the pretraining experiment pipeline.

    Args:
        config: An ExperimentConfig instance.
    """

    try:
        config = setup(config)

        logger = config.logger

        run_experiments(config)

        logger.info("-" * 40)
        logger.info(f"Finished. See {config.res_dir}")

    finally:
        barrier()
        torch._dynamo.reset()
        torch.cuda.empty_cache()
