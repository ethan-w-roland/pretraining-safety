"""Mixed-batch gradient routing for the slice-partitioned MoE MLP."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.model.config import RoutedModelConfig
from src.model.routed import RoutedTransformer
from src.run.util.config import ExperimentConfig, StageConfig


@dataclass
class RoutingConfig(StageConfig):
    """Train on all labels; route each sequence's MLP gradients to its expert.

    Core is always active in forward. Auxiliary sequences additionally activate
    their own expert, and only that expert receives their MLP parameter-gradient
    contributions. The shared attention/embedding/norm/output trunk trains
    only on core sequences.
    """

    model: RoutedModelConfig
    name: str = "routing"


def routing_masks(labels: torch.Tensor, num_experts: int) -> dict[str, torch.Tensor]:
    """labels: (B,) expert indices -> boolean fwd_mask/bck_mask: (B, K).

    Retained sequences: fwd=core, bck=core.
    Auxiliary sequences: fwd=core+their aux expert, bck=their aux expert.
    The model broadcasts these masks across each sequence's token positions.
    """
    bck_mask = F.one_hot(labels, num_classes=num_experts).bool()
    fwd_mask = bck_mask.clone()
    fwd_mask[:, 0] = True # core always active in forward
    return {"fwd_mask": fwd_mask, "bck_mask": bck_mask}


def do_routing(
    stage: RoutingConfig,
    model: RoutedTransformer,
    config: ExperimentConfig,
) -> RoutedTransformer:
    """Train each mixed loader batch in one forward with per-sequence masks."""
    from src.run.train.base import do_train
    return do_train(
        stage=stage,
        model=model,
        config=config,
        train_labels=config.data.labels,
        eval_expert_configs=[["core"]],
    )


def expert_configurations(labels: list[str]) -> list[list[str]]:
    """Evaluate core alone, then core with each auxiliary expert separately."""
    return [["core"]] + [["core", label] for label in labels[1:]]
