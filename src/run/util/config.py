"""
Run configuration and setup.

This module is the single entry point for initialising everything a training
run needs: CUDA device, tokenizer, shard paths, model config, results
directory, and logging.  The ``setup()`` function is called once at the start
of ``run()`` in main.py and returns the filled-in ``ExperimentConfig``.

Every stage declares its own ``model``, ``train`` and ``eval`` configs, so batch
size, sequence length and token budget (``train.num_tokens``) can differ per
stage. Stages build loaders over any set of labels with ``make_loader()``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import warnings
import torch

from dotenv import load_dotenv
from dataclasses import dataclass, field
from pathlib import Path
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from src.model.config import ModelConfig
from src.run.util.data import make_datasets
from src.run.util.tools import get_timestamp, set_seeds, json_safe
from src.run.util.logger import setup_logger
from src.run.util.distributed import (
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    broadcast_object,
    is_distributed,
    setup_distributed,
)

# --------------------------------------------------------------------------- #
# global log suppression for TorchDynamo recompilation warnings               #
# --------------------------------------------------------------------------- #

# Silence TorchDynamo recompilation warnings without setting invalid TORCH_LOGS
warnings.filterwarnings(
    "ignore",
    message=r".*torch\._dynamo.*recompile_limit.*",
    category=UserWarning,
)

# Silence Dynamo warnings about DDP's _broadcast_coalesced (can't trace through DDP internals)
warnings.filterwarnings(
    "ignore",
    message=r".*_broadcast_coalesced.*",
    category=UserWarning,
)

# Reduce logger verbosity for torch._dynamo in the current process
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()


# --------------------------------------------------------------------------- #
# dataclasses                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    """Configuration for training."""
    lr: float = 1e-4
    bs: int = 128
    ctx_len: int = 512
    acc_steps: int = 1 # effective batch size = bs * acc_steps * num_gpus
    warmup_prc: float = 0.1
    decay_prc: float = 0.1
    epochs: int = 1
    adam_betas: tuple[float, float] = (0.9, 0.95)
    seed: int = 42
    num_tokens: int = -1 #total tokens to train over, -1 for all data

@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    run: bool = True #do eval on test set?
    sample: bool = False #generate samples?
    elicit: bool = True #do adversarial ft?
    num_train_evals: int = 0 #number of validation evals during training

@dataclass
class StageConfig:
    """Base configuration shared by all stage types."""
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    name: str = "" # unique per experiment; also the stage's results dir name
    res_dir: Path = field(default_factory=Path)

@dataclass
class DataConfig:
    """Configuration for data limits."""
    dirs: list[Path] = field(default_factory=list)
    core: list[str] = field(default_factory=list)
    aux: list[str] = field(default_factory=list)
    datasets: dict = field(default_factory=dict) # label -> split -> shards
    @property
    def labels(self) -> list[str]:
        return ["core"] + sorted(self.aux)

@dataclass
class ExperimentConfig:
    """Shared configuration for runtime over all stages."""
    stages: list[StageConfig] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    compile: bool = True
    find_unused_parameters: bool = False
    is_ddp: bool = False
    log_level: str = "INFO"
    num_gpus: int = 1
    process_id: int = -1
    device: torch.device = torch.device("cuda")
    logger: logging.Logger = field(default_factory=logging.getLogger)
    res_root: Path = field(default_factory=Path)
    experiment_id: str = field(default_factory=get_timestamp)
    @property
    def res_dir(self) -> Path:
        return self.res_root / self.experiment_id

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def validate_stages(stages: list[StageConfig]) -> None:
    
    assert len(stages) > 0, "at least one stage is required"
    stage_names = [s.name for s in stages]
    assert len(set(stage_names)) == len(stage_names), f"Stage names must be unique: {stage_names}"

    # rotary embeddings are only built up to the model's ctx_len
    for stage in stages:
        assert stage.train.ctx_len <= stage.model.ctx_len, (
            f"stage {stage.name}: train.ctx_len {stage.train.ctx_len} > model.ctx_len {stage.model.ctx_len}"
        )


def setup_tokenizer(metadata_path: Path, logger: logging.Logger) -> AutoTokenizer:
    """Setup tokenizer from metadata."""
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {metadata_path}. Please run data prep.")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    tokenizer_name = metadata["all"].get("tokenizer")
    if tokenizer_name is None:
        tokenizer_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    vocab_size = len(tokenizer)
    logger.info(f"Tokenizer vocabulary: {vocab_size}")

    metadata_vocab_size = metadata["all"].get("vocab_size")
    if metadata_vocab_size is not None:
        assert vocab_size == metadata_vocab_size, f"Vocab size mismatch: tokenizer={vocab_size}, metadata={metadata_vocab_size}"
    else:
        logger.warning(f"No vocab_size in metadata.json — skipping validation (tokenizer has {vocab_size})")

    return tokenizer


def get_git_info() -> tuple[str, str]:
    """Return (branch, commit) for the repo containing this file.

    Falls back to "unknown" for either field if git is unavailable or the
    working directory is not a git repo (e.g. shipped as a tarball).
    """
    repo_dir = Path(__file__).resolve().parent
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=repo_dir,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return "unknown"

    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run(["rev-parse", "HEAD"])
    return branch, commit


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #


def setup(config: ExperimentConfig) -> ExperimentConfig:
    """
    Takes a partially filled ExperimentConfig, fills missing values, and initializes environment.
    """

    load_dotenv(override=True)
    setup_distributed()
    set_seeds(42)

    config.data.dirs = [Path(d) for d in config.data.dirs]
    assert len(config.data.dirs) > 0, "config.data.dirs must be provided"

    validate_stages(config.stages)

    # Synchronize experiment ID across ranks (timestamps differ per process)
    config.experiment_id = broadcast_object(config.experiment_id, src=0)

    # CUDA setup
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    config.device = device
    config.is_ddp = is_distributed()

    # Only create directory on main process to avoid duplicates in ddp
    if is_main_process():
        config.res_dir.mkdir(parents=True, exist_ok=True)
    barrier() # wait for creation

    # Define the stage directories
    for stage in config.stages:
        stage.res_dir = config.res_dir / stage.name

    # Setup logger
    config.process_id = get_rank()
    log_file = config.res_dir / "training.log"
    logger = setup_logger(
        name=f"training_{config.experiment_id}",
        log_file=log_file,
        level=config.log_level,
        process_id=config.process_id,
    )
    config.logger = logger

    branch, commit = get_git_info()
    logger.info(f"Git branch: {branch}, commit: {commit}")

    # Get number of GPUs
    config.num_gpus = get_world_size()
    logger.info(f"Number of GPUs: {config.num_gpus}")

    # Setup tokenizer
    dataset_metadata_path = config.data.dirs[0] / "metadata.json"
    tokenizer = setup_tokenizer(dataset_metadata_path, logger)

    # Round vocab size to nearest multiple of 64
    vocab_size = 64 * ((len(tokenizer) + 63) // 64)
    logger.info(f"Set model vocab size to nearest multiple of 64 of tokenizer size: {vocab_size}")

    # Update model config for each stage
    for stage in config.stages:
        assert isinstance(stage.model, ModelConfig), f"Stage {stage.res_dir.name} must declare a model config"
        stage.model.tokenizer = tokenizer
        stage.model.vocab_size = vocab_size
  
    # Load datasets
    config.data.datasets = make_datasets(config.data.dirs, config.data.core)

    # Save configuration info
    if is_main_process():
        config_dump = json_safe(config)
        config_text = json.dumps(config_dump, indent=4)
        config_path = config.res_dir / "config.json"
        config_path.write_text(config_text)
        logger.debug(f"Saved config to {config_path}")
        logger.debug(config_text)

    return config