"""
Distillation-time data filtering.

A fresh student is trained to match a frozen teacher's next-token distribution
(forward KL) on only the retained data. The teacher is either the baseline,
the filtering-stage model for the same retain target, or the baseline with a
specialist's excess logits subtracted (``teacher_type="difference"``):

    z = z_teacher - alpha * ReLU(z_specialist - z_teacher)

where the specialist is the baseline fine-tuned on the removed labels.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.model.config import Transformer
from src.model.utils import save_model
from src.run.eval import eval_loss
from src.run.train.base import make_scheduler
from src.run.util.config import ExperimentConfig, StageConfig, TrainConfig
from src.run.util.data import make_loader
from src.run.util.distributed import barrier, get_raw_model, is_main_process
from src.run.util.logger import get_tqdm_kwargs
from src.run.util.tools import save_losses, set_seeds


@dataclass
class DistillConfig(StageConfig):
    """Train a fresh student per retain target on KL to a frozen teacher."""
    name: str = "distill"
    retain_targets: list[list[str]] | None = None
    teacher: str = "" # name of the earlier stage whose checkpoint is the teacher
    teacher_type: Literal["baseline", "filtering", "difference"] = "baseline"
    specialist: TrainConfig = field(default_factory=TrainConfig) # "difference" only: fine-tuning the teacher on the removed labels
    alpha: float = 1.0
    temperature: float = 1.0


@torch.no_grad()
def teacher_log_probs(
    teacher: Transformer,
    anti_teacher: Optional[Transformer],
    x: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    """Teacher log-probs at ``temperature``, minus the anti-teacher's excess logits if given."""

    logits = teacher(x)[0].float()
    if anti_teacher is not None:
        anti_logits = anti_teacher(x)[0].float()
        logits = logits - alpha * F.relu(anti_logits - logits)
    return F.log_softmax(logits / temperature, dim=-1)


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logp: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Per-token forward KL(teacher || student).

    Scaled by ``temperature**2`` so gradient size does not depend on the temperature.
    """

    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    vocab_size = student_logp.size(-1)
    kl = F.kl_div(
        student_logp.view(-1, vocab_size),
        teacher_logp.view(-1, vocab_size),
        log_target=True,
        reduction="batchmean",
    )
    return kl * temperature ** 2


def do_distill(
    stage: DistillConfig,
    model: Transformer,
    config: ExperimentConfig,
    train_labels: Iterable[str],
    teacher: Transformer,
    anti_teacher: Optional[Transformer] = None,
) -> Transformer:
    """
    Train ``model`` on ``train_labels`` to match ``teacher``, then save the
    final checkpoint and loss histories to ``stage.res_dir``.

    Returns:
        Trained student
    """

    # unpack experiment config
    logger = config.logger
    is_ddp = config.is_ddp

    # unpack stage config
    train = stage.train
    acc_steps = train.acc_steps
    num_evals = stage.eval.num_train_evals
    alpha = stage.alpha
    temperature = stage.temperature
    train_labels = list(train_labels)

    # rotary embeddings are only built up to each model's ctx_len
    for frozen in (teacher, anti_teacher):
        if frozen is not None:
            frozen_ctx_len = get_raw_model(frozen).config.ctx_len
            assert frozen_ctx_len >= train.ctx_len, (
                f"teacher ctx_len {frozen_ctx_len} < student train.ctx_len {train.ctx_len}"
            )

    logger.info(
        f"---- Begin Distill | Train Labels: {train_labels} | "
        f"anti-teacher: {anti_teacher is not None} | alpha: {alpha} | T: {temperature} ----"
    )

    model.train()
    set_seeds(train.seed)

    loader = make_loader(config, stage, labels=train_labels, split="train")

    # calculate total steps
    num_total_steps = (len(loader) // acc_steps) * train.epochs
    logger.info(f"num_total_steps: {num_total_steps}")
    eval_freq = max(1, round(num_total_steps / num_evals)) if num_evals > 0 else -1

    opt = torch.optim.AdamW(model.parameters(), lr=train.lr, fused=True, betas=train.adam_betas)
    scheduler = make_scheduler(stage, opt, num_total_steps, logger)
    cur_lr = scheduler.get_last_lr()[0]

    losses = {"train": [], "val": {label: [] for label in config.data.labels}} # train: (step, acc_idx, kl)

    pbar = tqdm(total=num_total_steps, **get_tqdm_kwargs(logger, ncols=150))

    step = 0
    for epoch_idx in range(train.epochs):

        loader.reset(epoch_idx)

        for batch_idx, (_, x, _) in enumerate(loader):

            acc_idx = batch_idx % acc_steps
            is_last_acc = acc_idx == acc_steps - 1
            no_sync_ctx = model.no_sync() if (is_ddp and not is_last_acc) else contextlib.nullcontext()

            teacher_logp = teacher_log_probs(teacher, anti_teacher, x, alpha, temperature)

            with no_sync_ctx:
                student_logits = model(tokens=x)[0]
                loss = kd_loss(student_logits, teacher_logp, temperature)
                (loss / acc_steps).backward()

            cur_loss = loss.item()
            losses["train"].append((step, acc_idx, cur_loss))
            pbar.set_description(f"AC {acc_idx + 1}/{acc_steps} | LR: {cur_lr:.2e} | KL: {cur_loss:.3f}")

            if not is_last_acc:
                continue

            step += 1
            pbar.update()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()
            cur_lr = scheduler.get_last_lr()[0]

            # student CE on each label's test split
            if eval_freq > 0 and step % eval_freq == 0:
                for label in config.data.labels:
                    val_loss = eval_loss(model, config, stage, data_label=label, num_batches=50, shuffle_seed=step)
                    losses["val"][label].append((step, val_loss))

            if (step == 1) or (step % 1000 == 0) or (step == num_total_steps):
                avg_kl = np.mean([entry[-1] for entry in losses["train"][-10 * acc_steps:]])
                val_str = " ".join(
                    f"{label.upper()}: {vals[-1][1]:.2f}" for label, vals in losses["val"].items() if vals
                )
                logger.info(f"Step: {step}, LR: {cur_lr:.2e}, Train KL: {avg_kl:.3f}, Val CE: {val_str}")

        barrier()

    pbar.close()

    save_model(stage.res_dir, model)
    save_losses(stage.res_dir, losses)

    return model
