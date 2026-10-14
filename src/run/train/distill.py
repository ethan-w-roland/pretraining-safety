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

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import torch
import torch.nn.functional as F

from src.model.config import Transformer
from src.run.train.base import train_loop
from src.run.util.config import ExperimentConfig, StageConfig, TrainConfig
from src.run.util.distributed import get_raw_model


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
        logits = logits - alpha * (anti_logits - logits)
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
    """Train ``model`` on ``train_labels`` to match ``teacher`` (see ``train_loop``)."""

    # rotary embeddings are only built up to each model's ctx_len
    for frozen in (teacher, anti_teacher):
        if frozen is not None:
            frozen_ctx_len = get_raw_model(frozen).config.ctx_len
            assert frozen_ctx_len >= stage.train.ctx_len, (
                f"teacher ctx_len {frozen_ctx_len} < student train.ctx_len {stage.train.ctx_len}"
            )

    config.logger.info(
        f"Distill | anti-teacher: {anti_teacher is not None} | alpha: {stage.alpha} | T: {stage.temperature}"
    )

    def loss_fn(labels, x, y):
        teacher_logp = teacher_log_probs(teacher, anti_teacher, x, stage.alpha, stage.temperature)
        student_logits = model(tokens=x)[0]
        return kd_loss(student_logits, teacher_logp, stage.temperature)

    return train_loop(stage, model, config, train_labels, loss_fn, loss_name="KL")
