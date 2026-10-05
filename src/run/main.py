"""
export OMP_NUM_THREADS=16 && torchrun --nproc_per_node=8 -m src.run.main

main.py — multiclass-gradient-routing pipeline.

Architecture
------------
The pipeline runs a sequence of *stages*, each producing checkpoints in its
own subdirectory under the results root.  Stages are defined as plain dicts
in a ``stages`` list (see the CLI block at the bottom for examples).

Execution flow::

    run()                   # entry point: sets up DDP, calls setup()
      -> run_experiments()  # iterates over stages, dispatches to runners
          -> run_baseline   # train from scratch on all data
          -> run_filtering  # retrain without target data (per target group)
          -> run_distill    # KL-distill a fresh student from a frozen teacher on retained data

Each stage runner is self-contained: it creates the model and delegates
training to the appropriate ``do_*`` function, which writes the final
``checkpoint.pth``. Every run trains all of its stages from scratch into a
fresh results directory; later stages may load earlier stages' checkpoints.

Adversarial fine-tuning
-----------------------
Several stages optionally run "adversarial fine-tuning" (``ft_forget=True``):
after the main training, the model is fine-tuned on each forget-target label
individually and evaluated to measure how easily the model can re-learn the
information.  This is an *elicitation* metric.
"""

import dataclasses
import gc
import json
from pathlib import Path
from typing import Callable, Optional
import warnings
import torch

from src.model.base import BaseTransformer
from src.model.config import Transformer
from src.model.utils import copy_model, load_model, make_model
from src.run.eval import do_eval
from src.run.util.tools import json_safe, labels_to_str
from src.run.util.config import ExperimentConfig, StageConfig, setup
from src.run.util.distributed import barrier, get_raw_model

from src.run.train.base import BaselineConfig, FilteringConfig, do_train
from src.run.train.elicit import do_elicit
from src.run.train.distill import DistillConfig, do_distill
warnings.filterwarnings("ignore", message=r"(?s).*Online softmax is disabled.*", category=UserWarning)

def run_experiment(
    stage: StageConfig,
    model: Transformer,
    config: ExperimentConfig,
    func: Callable,
    func_args: Optional[dict] = None,
    eval_configs: Optional[list[dict]] = None,
) -> Transformer:
    """
    Modify a model in some way, then evaluate the effects of the modification.

    Args:
        stage: Typed stage configuration.
        model: Model to train (may be DDP-wrapped).
        config: Run-level configuration (loaders, device, logger, etc.).
        func: Training function (e.g. do_train, do_distill).
        func_args: Extra kwargs forwarded to *func*.
        eval_configs: List of eval kwarg dicts.  Each dict is unpacked into
            do_eval as a separate call.  Defaults to a single empty-dict call.

    Returns:
        The model after training.
    """
    if func_args is None:
        func_args = dict()
    if eval_configs is None:
        eval_configs = [{}]

    model = func(
        stage=stage,
        model=model,
        config=config,
        **func_args,
    )

    for ec in eval_configs:
        do_eval(
            stage=stage,
            model=model,
            config=config,
            **ec,
        )

    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_elicitation(
    stage: StageConfig,
    cur_dir: Path | None,
    model: Transformer,
    config: ExperimentConfig,
    data_labels: set[str] | list[str],
    expert_labels: list[str] | None = None,
    log_extra: dict | None = None,
) -> None:
    """Elicitation: copy model per label, finetune, eval, cleanup."""

    model = get_raw_model(model).to("cpu", dtype=torch.bfloat16)
    torch.cuda.empty_cache()

    save_dir = cur_dir / "elicit"
    save_dir.mkdir(parents=True, exist_ok=True)

    for label in sorted(data_labels):

        ft_model = copy_model(model, config)

        log_args = {"finetune": label, "elicited": True}
        if log_extra:
            log_args.update(log_extra)

        do_elicit(
            stage=stage,
            model=ft_model,
            config=config,
            data_labels=[label],
            expert_labels=expert_labels,
            log_args=log_args,
        )

        del ft_model
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    return None


def get_retain_targets(aux_labels: list[str]) -> list[list[str]]:
    """Get the list of retain targets for the auxiliary labels."""
    if len(aux_labels) == 1:
        return [["core"]]
    else:
        return [["core"]] + [["core", x] for x in aux_labels]
    

# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_baseline(
    stage: BaselineConfig,
    config: ExperimentConfig
) -> Transformer:
    """Train the baseline model. Returns the model on CPU."""

    logger = config.logger
    labels = config.data.labels

    logger.info("BASELINE START")

    model = make_model(BaseTransformer, stage.model, config)

    model = run_experiment(
        stage=stage,
        model=model,
        config=config,
        func=do_train,
        func_args={"train_labels": sorted(labels)},
        eval_configs=[{"log": {
            "retained": sorted(labels),
            "finetune": None,
            "elicited": False,
        }}],
    )

    if stage.eval.elicit:

        logger.info("BASELINE - ADVERSARIAL FT START")
        run_elicitation(
            stage=stage,
            cur_dir=stage.res_dir,
            model=model,
            config=config,
            data_labels=config.data.aux,
            log_extra={"retained": sorted(labels)})

    barrier()

    return model


def run_filtering(
    stage: FilteringConfig,
    config: ExperimentConfig,
) -> None:
    """Data-filtering stage: retrain from scratch on only the retained labels."""

    logger = config.logger
    labels = config.data.labels

    logger.info("FILTERING START")

    retain_targets = get_retain_targets(config.data.aux) # core & all core + 1 labels, less all labels
    if stage.retain_targets is not None:
        retain_targets = stage.retain_targets
        logger.info(f"Using stage-level retain_targets override ({len(retain_targets)} targets)")

    for retained in retain_targets:

        removed = sorted(set(labels) - set(retained))
        if not removed:
            continue

        iter_name = labels_to_str(retained)

        logger.info(f"Filtering: retaining {retained}, removing {removed}")

        model = make_model(BaseTransformer, stage.model, config)

        iter_dir = stage.res_dir / iter_name
        iter_stage = dataclasses.replace(stage, res_dir=iter_dir)

        model = run_experiment(
            stage=iter_stage,
            model=model,
            config=config,
            func=do_train,
            func_args={"train_labels": retained},
            eval_configs=[{"log": {
                "retained": sorted(retained),
                "finetune": None,
                "elicited": False
            }}],
        )

        if stage.eval.elicit:

            logger.info("FILTERING - ADVERSARIAL FT START")
            run_elicitation(
                stage=stage,
                cur_dir=iter_dir,
                model=model,
                config=config,
                data_labels=removed,
                log_extra={"retained": sorted(retained)},
            )

        barrier()


def run_distill(
    stage: DistillConfig,
    config: ExperimentConfig,
) -> None:
    """Distillation stage: per retain target, train a fresh student on KL to a
    frozen teacher over only the retained data.
    """

    logger = config.logger
    labels = config.data.labels

    logger.info("DISTILL START")

    retain_targets = get_retain_targets(config.data.aux)
    if stage.retain_targets is not None:
        retain_targets = stage.retain_targets

    teacher_dir = config.res_dir / stage.teacher

    for retained in retain_targets:

        removed = sorted(set(labels) - set(retained))
        iter_name = labels_to_str(retained)

        logger.info(f"Distill: retaining {retained}, removing {removed}")

        iter_dir = stage.res_dir / iter_name
        iter_stage = dataclasses.replace(stage, res_dir=iter_dir)

        anti_teacher = None

        if stage.teacher_type == "baseline":
            teacher = load_model(teacher_dir, config, frozen=True)

        elif stage.teacher_type == "filtering":
            teacher = load_model(teacher_dir / iter_name, config, frozen=True)

        elif stage.teacher_type == "difference":
            logger.info(f"Distill: fine-tuning specialist on {removed}")
            spec_stage = dataclasses.replace(
                stage,
                train=stage.specialist,
                eval=dataclasses.replace(stage.eval, num_train_evals=0),
                res_dir=iter_dir / "specialist",
            )
            specialist = load_model(teacher_dir, config)
            specialist = do_train(spec_stage, specialist, config, train_labels=removed)
            del specialist
            torch.cuda.empty_cache()

            teacher = load_model(teacher_dir, config, frozen=True)
            anti_teacher = load_model(spec_stage.res_dir, config, frozen=True)

        else:
            raise ValueError(f"Unknown teacher_type: {stage.teacher_type}")

        model = make_model(BaseTransformer, stage.model, config)

        model = run_experiment(
            stage=iter_stage,
            model=model,
            config=config,
            func=do_distill,
            func_args={
                "train_labels": retained,
                "teacher": teacher,
                "anti_teacher": anti_teacher,
                },
            eval_configs=[{"log": {
                "retained": sorted(retained),
                "finetune": None,
                "elicited": False
            }}],
        )

        del teacher, anti_teacher
        gc.collect()
        torch.cuda.empty_cache()

        if stage.eval.elicit and removed:
            logger.info("DISTILL - ADVERSARIAL FT START")
            run_elicitation(
                stage=stage,
                cur_dir=iter_dir,
                model=model,
                config=config,
                data_labels=removed,
                log_extra={"retained": sorted(retained)},
            )

        del model
        torch.cuda.empty_cache()
        barrier()

    gc.collect()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Pipeline dispatcher
# ---------------------------------------------------------------------------

RUNNERS = {
    BaselineConfig: run_baseline,
    FilteringConfig: run_filtering,
    DistillConfig: run_distill,
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


def run(config: ExperimentConfig):
    """Run the pretraining experiment pipeline.

    Args:
        config: An ExperimentConfig instance.

    Returns:
        DataFrame of eval stats (empty if no stats were written).
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

