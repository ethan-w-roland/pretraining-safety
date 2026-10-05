"""
Quick end-to-end run of every stage in run.py at 1 token per parameter
(~23M tokens for 20M, ~6M for 5M), printing a loss table at the end.

uv run torchrun --nproc_per_node=2 -m src.run.experiment.distill.test
"""

import dataclasses
import json

from src.run.experiment.distill.run import make_config, root_dir
from src.run.main import run
from src.run.util.distributed import is_main_process

config = make_config(seed=1, tokens_per_param=1)
config.res_root = root_dir.parent / "results" / "distill_test"
for stage in config.stages:
    stage.eval = dataclasses.replace(stage.eval, num_train_evals=10)

run(config)

if is_main_process():
    forget = config.data.aux[0]
    columns = {
        ("do_eval", "core"): "core",
        ("do_eval", forget): "forget",
        ("do_elicit", forget): "forget elicited",
    }

    losses = {stage.name: {} for stage in config.stages}
    for line in open(config.res_dir / "stats.jsonl"):
        row = json.loads(line)
        column = columns[(row["function"], row["data_label"])]
        losses[row["stage"]["name"]][column] = row["loss"]

    print(f"\nResults in {config.res_dir}\n")
    print(f"{'stage':>18}" + "".join(f"{column:>17}" for column in columns.values()))
    for name, stage_losses in losses.items():
        cells = [f"{stage_losses[column]:.3f}" if column in stage_losses else "-" for column in columns.values()]
        print(f"{name:>18}" + "".join(f"{cell:>17}" for cell in cells))
