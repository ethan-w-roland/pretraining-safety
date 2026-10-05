"""
Standard (non-routed) training loop.

This is the simplest training loop in the pipeline — a single optimizer over
all model parameters, no expert routing, no per-label gradient control.  Used
for baseline pretraining and the data-filtering stage (retrain from scratch
on a subset of labels).

Key design points:

- **Single AdamW optimizer** with a three-phase LR schedule:
  10% warmup (linear ramp), 80% constant, 10% linear decay.
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
from src.run.util.distributed import is_main_process, barrier
from src.model.utils import save_model
from src.run.eval import eval_loss


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
    lr = stage.train.lr
    end_factor = 1e-8 / lr
    warmup_steps = round(warmup_prc * num_total_steps)
    decay_steps = round(decay_prc * num_total_steps)
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
            decay_progress = (current_step - warmup_steps - constant_steps) / decay_steps
            return 1.0 - (1.0 - end_factor) * decay_progress

    return LambdaLR(opt, lr_lambda)


def do_train(
    stage: StageConfig,
    model: BaseTransformer,
    config: ExperimentConfig,
    train_labels: Iterable[str],
) -> BaseTransformer:
    """
    Train a transformer model on specified train data labels, then save the
    final checkpoint and loss histories to ``stage.res_dir``.

    Args:
        stage: Stage configuration
        model: Model to train
        config: Run configuration
        train_labels: Data labels to train on
    
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
    adam_betas = stage.train.adam_betas
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
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        fused=True, 
        betas=adam_betas)

    # setup losses
    losses = {"train": [], "val": {}} # train: (step, acc_idx, loss)
    for label in config.data.labels:
        losses["val"][label] = [] #(step, loss)

    # setup scheduler
    scheduler = make_scheduler(stage, opt, num_total_steps, logger)
    cur_lr = scheduler.get_last_lr()[0]

    # setup progress bar
    pbar = tqdm(total = num_total_steps, **get_tqdm_kwargs(logger, ncols=150))

    # training loop
    step = 0
    for epoch_idx in range(epochs):

        loader.reset(epoch_idx)

        for batch_idx, batch in enumerate(loader):

            _, x, y = batch
            acc_idx = batch_idx % acc_steps
            is_last_acc = acc_idx == acc_steps - 1
            no_sync_ctx = model.no_sync() if (is_ddp and not is_last_acc) else contextlib.nullcontext()
            
            with no_sync_ctx:
                _, loss = model.forward(
                    tokens=x,
                    targets=y,
                )
                scaled_loss = loss / acc_steps
                scaled_loss.backward()

            # train loss logging
            cur_loss = loss.item()
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
                opt.step()
                opt.zero_grad(set_to_none=True)

                # update learning rate
                scheduler.step()
                cur_lr = scheduler.get_last_lr()[0]

                # measure validation loss
                if eval_freq > 0 and step % eval_freq == 0:
                    for label in config.data.labels:
                        val_loss = eval_loss(
                            model, config, stage,
                            data_label=label, 
                            num_batches=50, 
                            shuffle_seed=step)
                        losses["val"][label].append((step, val_loss))

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