from __future__ import annotations
import itertools
import numpy as np
import torch
from typing import Any, Iterable, Optional
from tqdm import tqdm

from src.run.util.config import ExperimentConfig, StageConfig
from src.model.config import Transformer
from src.run.util.data import make_loader
from src.run.util.tools import log_line, get_exp_mask, json_safe
from src.run.util.logger import get_tqdm_kwargs
from src.run.util.distributed import get_raw_model, is_main_process, barrier, reduce_tensor

# --------------------------------------------------------------------------- #
# EVAL LOSS                                                                   #
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def eval_loss(
    model: Transformer,
    config: ExperimentConfig,
    stage: StageConfig,
    data_label: str,
    expert_labels: Optional[Iterable[str]] = None,
    num_batches: Optional[int] = None,
    split: str = "test",
    shuffle_seed: int = 0,
) -> float:
    """Mean loss over ``data_label`` seqs, batched and sized as in ``stage.train``."""

    # Ensure all GPUs are synchronized before starting evaluation
    barrier()

    # unpack config
    logger = config.logger
    labels = config.data.labels
    device = config.device

    loader = make_loader(config, stage, [data_label], split)
    loader.reset(shuffle_seed)

    # Get raw model for accessing config and model_type
    raw_model = get_raw_model(model)
    model_type = type(raw_model).__name__

    total_loss = 0.0
    max_batches = len(loader)
    if num_batches is None:
        num_batches = max_batches
    num_batches = min(num_batches, max_batches)
    
    # Guard against empty loaders (e.g., test split smaller than batch size)
    if num_batches == 0:
        logger.warning(f"No batches found for label={data_label}; returning NaN loss")
        return float('nan')

    was_training = model.training
    model.eval()

    for _, x, y in itertools.islice(loader, num_batches):

        if model_type in ("MoETransformer", "LoRATransformer", "DemixTransformer"):

            fwd_mask = get_exp_mask(
                labels=labels,
                selected_labels=expert_labels,
                device=device,
            )
            bck_mask = get_exp_mask(
                labels=labels,
                selected_labels=expert_labels,
                device=device,
            )
            loss = model(
                tokens=x,
                targets=y,
                fwd_mask=fwd_mask,
                bck_mask=bck_mask,
            )[1]
            
        else:

            loss = model(
                tokens=x,
                targets=y,
            )[1]

        total_loss += loss.item()

    model.train(was_training)

    loss = total_loss / num_batches
    loss = reduce_tensor(torch.tensor(loss, device=device)).item()
    return loss


# --------------------------------------------------------------------------- #
# GENERATION                                                                  #
# --------------------------------------------------------------------------- #

def collect_stories(
    config: ExperimentConfig,
    data_label: str,
    eos_id: int,
    num_stories: int,
    seed: int,
) -> list[list[int]]:
    """``num_stories`` random complete stories (EOS stripped) from the test shards of ``data_label``."""

    spans = []  # (tokens, start, end) of every story
    for path in config.data.datasets[data_label]["test"]:
        tokens = np.memmap(path, dtype=np.uint16, mode="r")
        ends = np.flatnonzero(tokens == eos_id)
        starts = np.concatenate([[0], ends[:-1] + 1])
        for start, end in zip(starts, ends):
            if end > start:
                spans.append((tokens, start, end))

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(spans), size=min(num_stories, len(spans)), replace=False)

    stories = []
    for i in chosen:
        tokens, start, end = spans[i]
        stories.append(tokens[start:end].tolist())
    return stories


@torch.inference_mode()
def generate_samples(
    model: Transformer,
    config: ExperimentConfig,
    stage: StageConfig,
    data_label: str,
    expert_labels: Optional[Iterable[str]] = None,
    num_examples: int = 128,
    prefix_len: int = 64,
) -> None:

    device = config.device
    logger = config.logger

    # Get raw model for accessing config and model_type
    raw_model = get_raw_model(model)

    def _generate_batch(batch_prompts: list[list[int]]) -> list[list[int]]:
        if not batch_prompts:
            return []

        eos_token_id = raw_model.config.eos_token_id
        bs = len(batch_prompts)
        max_prompt_len = max(len(p) for p in batch_prompts)

        # Right-align prompts; fill left with EOS to create clean segment boundaries
        prompt_batch = torch.full(
            (bs, max_prompt_len), eos_token_id, dtype=torch.long, device=device
        )
        for i, p in enumerate(batch_prompts):
            if len(p) > 0:
                prompt_batch[i, -len(p) :] = torch.tensor(p, dtype=torch.long, device=device)

        finished = [False] * bs
        generated_tok: list[list[int]] = [[] for _ in range(bs)]
        max_new_tokens = raw_model.config.ctx_len - max_prompt_len

        for _ in tqdm(range(max_new_tokens), **get_tqdm_kwargs(logger, desc=f"Generating samples | Data: {data_label}", ncols=100)):
            if raw_model.config.arch in ("moe", "lora", "demix"):
                labels = config.data.labels

                fwd_mask = get_exp_mask(labels, expert_labels, device=prompt_batch.device)
                bck_mask = get_exp_mask(labels, expert_labels, device=prompt_batch.device)

                logp = model(
                    tokens=prompt_batch,
                    targets=None,
                    fwd_mask=fwd_mask,
                    bck_mask=bck_mask,
                )[0]
                next_tok = torch.argmax(logp[:, -1, :], dim=-1)
            else:
                logits = model(prompt_batch)[0]
                next_tok = torch.argmax(logits[:, -1, :], dim=-1)

            prompt_batch = torch.cat([prompt_batch, next_tok.unsqueeze(1)], dim=1)

            for i, tok in enumerate(next_tok.tolist()):
                if finished[i]:
                    continue
                if tok == eos_token_id:
                    finished[i] = True
                else:
                    generated_tok[i].append(tok)
            if all(finished):
                break

        return generated_tok

    logger.info(f"Generating samples | Data: {data_label}")

    # 1. Collect complete sequences (strip trailing EOS)
    eos_id = raw_model.config.eos_token_id
    sequences_tok = collect_stories(config, data_label, eos_id, num_examples, stage.train.seed)
    if len(sequences_tok) == 0:
        return
    prompts_tok = [seq[:prefix_len] for seq in sequences_tok]

    # 2. Batched autoregressive generation
    gen_batch_size = stage.train.bs
    generations_tok: list[list[int]] = []
    for i in range(0, len(prompts_tok), gen_batch_size):
        batch_prompts = prompts_tok[i : i + gen_batch_size]
        generations_tok.extend(_generate_batch(batch_prompts))

    # 3. Decode and save
    tok = raw_model.config.tokenizer
    sequences_text = [tok.decode(s, skip_special_tokens=False) for s in sequences_tok]
    prompts_text = [tok.decode(p, skip_special_tokens=False) for p in prompts_tok]
    continuations_text = [tok.decode(g, skip_special_tokens=False) for g in generations_tok]
    generations = [
        {
            "prompt": prompts_text[i],
            "continuation": continuations_text[i],
            "truth": sequences_text[i],
        }
        for i in range(len(prompts_text))
    ]
        
    out = {
        "prefix_len": prefix_len,
        "num_examples": len(prompts_text),
        "generations": generations,
    }

    return out


# --------------------------------------------------------------------------- #
# MAIN                                                                       #
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def do_eval(
    stage: StageConfig,
    model: Transformer,
    config: ExperimentConfig,
    data_labels: Optional[Iterable[str]] = None,
    expert_labels: Optional[Iterable[str]] = None,
    log: Optional[dict[str, Any]] = None,
) -> None:

    if not stage.eval.run:
        return

    # unpack config
    gen_samples = stage.eval.sample
    logger = config.logger
    log_fp = config.res_dir / "stats.jsonl"

    if data_labels is None:
        data_labels = config.data.labels

    logger.info(f"---- Begin Eval ----")

    # evaluate for each label
    for data_label in data_labels:

        logger.info(f"Data: {data_label} | Experts: {expert_labels}")

        loss = eval_loss(model, config, stage, data_label, expert_labels)
        logger.info(f"loss: {loss:.8f}")

        # File I/O operations need guards
        if is_main_process():

            entry = {
                "stage": json_safe(stage),
                "function": "do_eval",
                "data_label": data_label,
                "expert_labels": expert_labels,
                "loss": loss,
            }

            if log:
                entry.update(log)

            # Generate samples (only on main process to avoid duplication)
            if gen_samples:
                samples = generate_samples(
                    model=model,
                    config=config,
                    stage=stage,
                    data_label=data_label,
                    expert_labels=expert_labels,
                )
                entry["samples"] = samples
            
            # Write to JSONL file
            log_line(entry, log_fp)