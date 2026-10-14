"""
The training loop shared by every stage, and the dense training stage.

``train_loop`` is the one optimizer loop in the pipeline: Muon + AdamW
(``PretrainingOptimizer``), warmup / constant / linear-decay LR, gradient
accumulation, periodic validation, then a final checkpoint and loss pickle
(train loss and per-label val loss histories, for offline analysis).
Stages differ only in the micro-batch loss they pass:

- ``do_train`` (baseline and data-filtering stages): next-token CE.
- ``do_routing`` (``src.run.train.routing``): CE with per-sequence expert masks.
- ``do_distill`` (``src.run.train.distill``): KL to a frozen teacher.

One loader covers all train labels, so every micro-batch mixes sequences from
all labels in proportion to their budgets.
"""

import contextlib
import numpy as np
import torch
import logging

from dataclasses import dataclass
from typing import Callable, Iterable
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR

from src.model.config import Transformer
from src.run.util.config import ExperimentConfig, StageConfig
from src.run.util.data import make_loader
from src.run.util.tools import set_seeds, save_losses
from src.run.util.logger import get_tqdm_kwargs
from src.run.util.distributed import barrier, is_distributed
from src.model.utils import save_model
from src.run.eval import eval_loss
from src.run.train.optim import PretrainingOptimizer

# (labels (B,), x (B, T), y (B, T)) -> scalar loss of one micro-batch, where
# labels are each sequence's index in config.data.labels
LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class BaselineConfig(StageConfig):
    """Configuration for the baseline (dense) training stage."""
    name: str = "baseline"


@dataclass
class FilteringConfig(StageConfig):
    """Configuration for the data-filtering evaluation stage."""
    name: str = "filtering"
    retain_targets: list[list[str]] | None = None


def make_scheduler(
    stage: StageConfig, 
    opt: torch.optim.Optimizer, 
    num_total_steps: int,
    logger: logging.Logger
) -> LambdaLR:

    warmup_prc = stage.train.warmup_prc
    decay_prc = stage.train.decay_prc
    end_factor = stage.train.min_lr_ratio
    assert 0 <= end_factor <= 1
    assert num_total_steps > 0, "training requires at least one optimizer step"
    warmup_steps = round(warmup_prc * num_total_steps)
    assert stage.train.warmup_steps >= 0
    warmup_steps = min(warmup_steps, stage.train.warmup_steps)
    decay_steps = round(decay_prc * num_total_steps)
    assert 0 <= warmup_steps <= num_total_steps
    assert 0 <= decay_prc <= 1 and 0 <= warmup_prc <= 1
    # A short smoke run still needs a final decay step, without dividing by 0.
    if decay_prc > 0:
        decay_steps = max(1, decay_steps)
        warmup_steps = min(warmup_steps, num_total_steps - decay_steps)
    constant_steps = num_total_steps - warmup_steps - decay_steps

    logger.info(f"warmup_steps: {warmup_steps}")
    logger.info(f"constant_steps: {constant_steps}")
    logger.info(f"decay_steps: {decay_steps}")

    def lr_lambda(current_step):

        if current_step < warmup_steps:
            if current_step == 0:
                logger.info("LR Scheduler: Warmup Start")
            return end_factor + (1.0 - end_factor) * (current_step / warmup_steps)

        elif current_step < warmup_steps + constant_steps:
            if current_step == warmup_steps:
                logger.info("LR Scheduler: Constant Start")
            return 1.0

        else:
            if current_step == warmup_steps + constant_steps:
                logger.info("LR Scheduler: Decay Start")
            if decay_steps == 0:
                return 1.0
            decay_progress = min(1.0, max(0.0, (current_step - warmup_steps - constant_steps) / decay_steps))
            return 1.0 - (1.0 - end_factor) * decay_progress

    return LambdaLR(opt, lr_lambda)


def train_loop(
    stage: StageConfig,
    model: Transformer,
    config: ExperimentConfig,
    train_labels: Iterable[str],
    loss_fn: LossFn,
    eval_expert_configs: list[list[str]] | None = None,
    loss_name: str = "CE",
) -> Transformer:
    """
    Train ``model`` on ``train_labels`` by minimising ``loss_fn``, then save the
    final checkpoint and loss histories to ``stage.res_dir``.

    Args:
        stage: Stage configuration
        model: Model to train
        config: Run configuration
        train_labels: Data labels to train on
        loss_fn: Loss of one micro-batch (see ``LossFn``)
        eval_expert_configs: Expert subsets to validate separately (routed
            models); None validates the whole model.
        loss_name: Name of the train loss in logs

    Returns:
        Trained model
    """

    # unpack config
    logger = config.logger
    train = stage.train
    acc_steps = train.acc_steps
    num_evals = stage.eval.num_train_evals
    train_labels = list(train_labels)

    logger.info(f"---- Begin Train | Train Labels: {train_labels} ----")

    model.train()
    set_seeds(train.seed)

    loader = make_loader(config, stage, labels=train_labels, split="train")

    # calculate total steps
    num_total_steps = (len(loader) // acc_steps) * train.epochs
    logger.info(f"num_total_steps: {num_total_steps}")
    eval_freq = max(1, round(num_total_steps / num_evals)) if num_evals > 0 else -1

    # setup optimizer and scheduler
    opt = PretrainingOptimizer(model, train)
    schedulers = [make_scheduler(stage, x, num_total_steps, logger) for x in opt.optimizers]
    cur_lr = schedulers[0].get_last_lr()[0]
    logger.info(f"Muon + AdamW: peak AdamW lr={train.adamw_lr}, peak Muon lr={train.muon_lr}")

    # setup losses
    losses = {"train": [], "val": {}} # train: (step, acc_idx, loss); val: key -> (step, loss)
    val_targets = [] # (data label, expert labels, val key)
    for experts in eval_expert_configs or [None]:
        for label in config.data.labels:
            key = label if experts is None else f"{label}|experts={'+'.join(experts)}"
            losses["val"][key] = []
            val_targets.append((label, experts, key))

    pbar = tqdm(total=num_total_steps, **get_tqdm_kwargs(logger, ncols=150))

    # training loop
    step = 0
    for epoch_idx in range(train.epochs):

        loader.reset(epoch_idx)

        for batch_idx, (labels, x, y) in enumerate(loader):

            acc_idx = batch_idx % acc_steps
            is_last_acc = acc_idx == acc_steps - 1
            no_sync_ctx = model.no_sync() if (is_distributed() and not is_last_acc) else contextlib.nullcontext()
            with no_sync_ctx:
                loss = loss_fn(labels, x, y)
                (loss / acc_steps).backward()
            opt.observe(labels)

            cur_loss = loss.item()
            losses["train"].append((step, acc_idx, cur_loss))
            pbar.set_description(f"AC {acc_idx + 1}/{acc_steps} | LR: {cur_lr:.2e} | {loss_name}: {cur_loss:.3f}")

            if not is_last_acc:
                continue

            step += 1
            pbar.update()

            # step the optimizer and learning rate
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()
            cur_lr = schedulers[0].get_last_lr()[0]

            # measure validation loss (CE on each label's test split)
            if eval_freq > 0 and step % eval_freq == 0:
                for label, experts, key in val_targets:
                    val_loss = eval_loss(
                        model, config, stage,
                        data_label=label,
                        expert_labels=experts,
                        num_batches=50,
                        shuffle_seed=step,
                    )
                    losses["val"][key].append((step, val_loss))

            if (step == 1) or (step % 1000 == 0) or (step == num_total_steps):
                avg_loss = np.mean([entry[-1] for entry in losses["train"][-10 * acc_steps:]])
                val_str = " ".join(
                    f"{key.upper()}: {vals[-1][1]:.2f}" for key, vals in losses["val"].items() if vals
                )
                logger.info(f"Step: {step}, LR: {cur_lr:.2e}, Train {loss_name}: {avg_loss:.3f}, Val CE: {val_str}")

        # sync after epoch
        barrier()

    pbar.close()

    save_model(stage.res_dir, model)
    save_losses(stage.res_dir, losses)

    return model


def do_train(
    stage: StageConfig,
    model: Transformer,
    config: ExperimentConfig,
    train_labels: Iterable[str],
) -> Transformer:
    """Train a dense model with next-token cross-entropy on ``train_labels``."""

    def loss_fn(labels, x, y):
        return model(tokens=x, targets=y)[1]

    return train_loop(stage, model, config, train_labels, loss_fn)
