"""
Logging utilities: rank-0-only logging that plays nicely with tqdm.

Under torchrun every rank calls ``setup_logger``; records from non-main ranks
are dropped and only rank 0 opens the log file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional
from tqdm.auto import tqdm
from src.run.util.distributed import get_rank, is_main_process


class DistributedFilter(logging.Filter):
    """Filter that only allows logging from main process in distributed training."""

    def filter(self, record):
        return is_main_process()


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that plays nicely with tqdm progress bars."""

    def emit(self, record):
        try:
            # Use tqdm.write to avoid breaking progress bars
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Set up a logger that writes to the console and, optionally, ``log_file``.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []
    logger.filters = []
    logger.addFilter(DistributedFilter())

    formatter = logging.Formatter(
        f'[GPU {get_rank()}] [%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S'
    )

    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None and is_main_process():
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_tqdm_kwargs(logger: logging.Logger, **kwargs: object) -> dict[str, object]:
    """
    Get tqdm kwargs based on logger level.

    Progress bars are only shown on the main process in DEBUG mode.
    In INFO mode or higher, progress bars are disabled entirely.
    """
    # Ensure tqdm writes to stderr to match logging handler
    kwargs.setdefault('file', sys.stderr)

    # Disable progress bars unless in DEBUG mode on main process
    kwargs['disable'] = not (logger.isEnabledFor(logging.DEBUG) and is_main_process())

    return kwargs
