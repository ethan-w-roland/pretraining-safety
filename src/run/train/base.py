"""
Standard training loop.

This is the simplest training loop in the pipeline — a single optimizer over
all model parameters, no expert routing, no per-label gradient control.  Used
for baseline pretraining and the data-filtering stage (retrain from scratch
on a subset of labels).

Key design points:

- **Muon + AdamW** for Trinity-style pretraining.
  Warmup, constant LR, then linear decay; durations are stage-configurable.
- **Loss pickle dumps**: the train loss and per-label val loss histories are
  saved alongside checkpoints for offline analysis and plotting.
- **Mixed batches**: one loader over all train labels, so every micro-batch
  mixes sequences from all labels in proportion to their budgets.
"""

import contextlib
import numpy as np
import torch
import logging

from dataclasses import dataclass
from typing import Iterable
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR

from src.model.base import BaseTransformer
from src.run.util.config import ExperimentConfig, StageConfig
from src.run.util.data import make_loader
from src.run.util.tools import set_seeds, save_losses
from src.run.util.logger import get_tqdm_kwargs
from src.run.util.distributed import barrier
from src.model.utils import save_model
from src.run.eval import eval_loss
from src.run.train.optim import PretrainingOptimizer
from src.run.train.routing import routing_masks
from src.run.util.distributed import get_raw_model


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


def do_train(
    stage: StageConfig,
    model: BaseTransformer,
    config: ExperimentConfig,
    train_labels: Iterable[str],
    eval_expert_configs: list[list[str]] | None = None,
) -> BaseTransformer:
    """
    Train a transformer model on specified train data labels, then save the
    final checkpoint and loss histories to ``stage.res_dir``.

    Args:
        stage: Stage configuration
        model: Model to train
        config: Run configuration
        train_labels: Data labels to train on
        eval_expert_configs: Optional expert subsets to validate separately.
    
    Returns:
        Trained model
    """

    # unpack experiment config
    logger = config.logger
    is_ddp = config.is_ddp

    #unpack stage config
    lr = stage.train.lr
    acc_steps = stage.train.acc_steps
    epochs = stage.train.epochs
    seed = stage.train.seed
    num_evals = stage.eval.num_train_evals
    res_dir = stage.res_dir

    logger.info(f"---- Begin Train | Train Labels: {train_labels} ----")

    model.train()
    set_seeds(seed)

    loader = make_loader(config, stage, labels=train_labels, split="train")

    # calculate total steps
    num_batches = len(loader)
    num_total_steps = (num_batches // acc_steps) * epochs
    logger.info(f"num_total_steps: {num_total_steps}")
    eval_freq = max(1, round(num_total_steps / num_evals)) if num_evals > 0 else -1

    # setup optimizer
    opt = PretrainingOptimizer(model, stage.train)
    optimizers = opt.optimizers

    # setup losses
    losses = {"train": [], "val": {}} # train: (step, acc_idx, loss)
    val_targets = []
    for experts in eval_expert_configs or [None]:
        for label in config.data.labels:
            key = label if experts is None else f"{label}|experts={'+'.join(experts)}"
            losses["val"][key] = [] # (step, loss)
            val_targets.append((label, experts, key))

    # setup scheduler
    schedulers = [make_scheduler(stage, optimizer, num_total_steps, logger) for optimizer in optimizers]
    cur_lr = schedulers[0].get_last_lr()[0]
    raw_model = get_raw_model(model)
    routed:bool = hasattr(raw_model, "labels")
    active_experts = torch.zeros(len(raw_model.labels), dtype=torch.int32, device=config.device) if routed else None
    logger.info(f"Muon + AdamW: peak AdamW lr={lr}, peak Muon lr={stage.train.muon_lr}")

    # setup progress bar
    pbar = tqdm(total = num_total_steps, **get_tqdm_kwargs(logger, ncols=150))

    # training loop
    step = 0
    for epoch_idx in range(epochs):

        loader.reset(epoch_idx)

        for batch_idx, batch in enumerate(loader):

            labels, x, y = batch
            if active_experts is not None:
                active_experts[labels.unique()] = 1
            acc_idx = batch_idx % acc_steps
            is_last_acc = acc_idx == acc_steps - 1
            kwargs = routing_masks(labels, len(raw_model.labels)) if routed else {}
            no_sync_ctx = model.no_sync() if (is_ddp and not is_last_acc) else contextlib.nullcontext()
            with no_sync_ctx:
                _, loss = model(tokens=x, targets=y, **kwargs)
                (loss / acc_steps).backward()
            cur_loss = loss.item()

            # train loss logging
            losses["train"].append((step, acc_idx, cur_loss))

            # update progress bar
            desc_str = f"AC {acc_idx + 1}/{acc_steps} | LR: {cur_lr:.2e} | L: {cur_loss:.2f}"
            pbar.set_description(desc_str)
            pbar.refresh()

            if is_last_acc:

                step += 1
                pbar.update()

                # step the optimizer
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if active_experts is not None and is_ddp:
                    torch.distributed.all_reduce(active_experts, op=torch.distributed.ReduceOp.MAX)
                opt.step(active_experts)
                if active_experts is not None:
                    active_experts.zero_()
                opt.zero_grad(set_to_none=True)

                # update learning rate
                for scheduler in schedulers:
                    scheduler.step()
                cur_lr = schedulers[0].get_last_lr()[0]

                # measure validation loss
                if eval_freq > 0 and step % eval_freq == 0:
                    for label, experts, key in val_targets:
                        val_loss = eval_loss(
                            model, config, stage,
                            data_label=label, 
                            expert_labels=experts,
                            num_batches=50, 
                            shuffle_seed=step)
                        losses["val"][key].append((step, val_loss))

                # logger printout
                if (step == 1) or (step % 1000 == 0) or (step == num_total_steps):

                    logger.info(f"Step: {step}, LR: {cur_lr:.2e}")

                    avg_train_loss = np.mean([x[-1] for x in losses["train"][-10 * acc_steps:]])
                    logger.info(f"Train Loss: {avg_train_loss:.2f}")

                    if eval_freq > 0:
                        val_loss_str = ""
                        for label in losses["val"].keys():
                            label_str = label.upper()
                            val_loss = losses["val"][label][-1][1] if len(losses["val"][label]) > 0 else float('nan')
                            val_loss_str += f"{label_str}: {val_loss:.2f} "
                        logger.info(f"Val Loss: {val_loss_str}")

        # sync after epoch
        barrier()

    # close progress bar
    pbar.close()

    save_model(res_dir, model)
    save_losses(res_dir, losses)

    return model
