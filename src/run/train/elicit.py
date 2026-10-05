"""
Adversarial fine-tuning for elicitation testing.

This is NOT a training stage that produces a model for downstream use.
Instead, it measures how easily a trained model can *re-learn* forgotten or
compartmentalized knowledge, which is the key metric for evaluating the
effectiveness of gradient routing and unlearning methods.

The procedure:
1. Copy the model being evaluated.
2. Take a fixed set of ``num_seq`` train seqs from the target labels.
3. Repeatedly take one full-batch gradient step on that set, measuring the
   target labels' test loss after each step, until it stops improving.
4. Log the best test loss to ``stats.jsonl``.
"""

from __future__ import annotations

import contextlib
import itertools
import torch
from typing import Any, Iterable, Optional
from tqdm.auto import tqdm

from src.model.config import Transformer
from src.run.eval import eval_loss
from src.run.util.config import ExperimentConfig, StageConfig
from src.run.util.data import make_loader
from src.run.util.tools import get_exp_mask, log_line, json_safe
from src.run.util.logger import get_tqdm_kwargs
from src.run.util.distributed import barrier, get_raw_model, get_world_size, is_main_process


def do_elicit(
    stage: StageConfig,
    model: Transformer,
    config: ExperimentConfig,
    data_labels: Iterable[str],
    expert_labels: Optional[Iterable[str]] = None,
    log_args: Optional[dict[str, Any]] = None,
    num_seq: int = 512,
    num_rounds: int = 200,
    patience: int = 10,
) -> Transformer:
    """
    Fine-tune ``model`` on ``num_seq`` seqs of ``data_labels`` and log the best
    test loss reached on each label.

    Returns:
        Finetuned model
    """

    logger = config.logger
    labels = config.data.labels
    log_fp = config.res_dir / "stats.jsonl"
    is_ddp = config.is_ddp
    lr = stage.train.lr / 4 # heuristic
    bs = stage.train.bs
    data_labels = list(data_labels)

    logger.info(f"---- Begin FT | Experts: {expert_labels} | Data: {data_labels} ----")
    logger.debug(f"num_seq: {num_seq}, num_rounds: {num_rounds}, patience: {patience}")

    assert all(label in labels for label in data_labels), f"all data labels must be in {labels}"

    raw_model = get_raw_model(model)
    model_type = type(raw_model).__name__
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)

    # each rank's share of the loader is disjoint, so together the ranks hold
    # num_seq distinct seqs; these same seqs are reused every round
    num_micro_batches = num_seq // (bs * get_world_size())
    assert num_micro_batches > 0, f"num_seq {num_seq} < bs {bs} x world size {get_world_size()}"
    loader = make_loader(config, stage, data_labels, "train", balance=True)
    micro_batches = [(x, y) for _, x, y in itertools.islice(loader, num_micro_batches)]

    logger.debug(f"{num_micro_batches} micro-batches of {bs} seqs per rank")

    patience_count = 0
    best_idx = 0
    losses = {lab: [] for lab in data_labels}

    pbar = tqdm(range(num_rounds), **get_tqdm_kwargs(logger, desc=f"FT", ncols=150))
    for round_idx in pbar:

        for batch_idx, (x, y) in enumerate(micro_batches):

            is_last = batch_idx == num_micro_batches - 1
            no_sync_ctx = model.no_sync() if (is_ddp and not is_last) else contextlib.nullcontext()

            with no_sync_ctx:
                if model_type in ("MoETransformer", "LoRATransformer", "DemixTransformer"):
                    fwd_mask = get_exp_mask(labels, expert_labels, device=x.device)
                    bck_mask = get_exp_mask(labels, expert_labels, device=x.device)
                    loss = model(x, targets=y, fwd_mask=fwd_mask, bck_mask=bck_mask)[1]
                else:
                    loss = model(x, targets=y)[1]

                (loss / num_micro_batches).backward()

            pbar.set_description(f"Round {round_idx} | Batch {batch_idx + 1}/{num_micro_batches} | L {loss.item():.4f}")

        opt.step()
        opt.zero_grad(set_to_none=True)

        for label in data_labels:
            val_loss = eval_loss(
                model,
                config,
                stage,
                data_label=label,
                expert_labels=expert_labels,
                num_batches=100,
            )
            losses[label].append(val_loss)

        # val losses are all-reduced, so every rank makes the same stopping decision
        last_losses = [losses[lab][-1] for lab in data_labels]
        val_loss = sum(last_losses) / len(last_losses)

        old_losses = [losses[lab][best_idx] for lab in data_labels]
        old_loss = sum(old_losses) / len(old_losses)

        if val_loss <= old_loss:
            best_idx = round_idx
            logger.info(f"New best validation loss: {val_loss:.4f} @ step {round_idx}")
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            logger.info(f"Stop FT @ {round_idx+1}: no val improvement for {patience} steps")
            break

    if is_main_process():
        for label in data_labels:
            loss = losses[label][best_idx]
            entry = {
                "stage": json_safe(stage),
                "function": "do_elicit",
                "data_label": label,
                "expert_labels": expert_labels,
                "loss": loss,
                "ft_step": best_idx,
                "ft_type": "single_batch",
            }
            if log_args:
                entry.update(log_args)
            log_line(entry, log_fp)
        logger.info(
            f"Wrote stats.jsonl entries at FT step {best_idx} "
        )

    barrier()

    return model
