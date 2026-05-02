from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
    logger.remove()
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add(sys.stderr, level=log_level, format=fmt, colorize=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(log_file), level=log_level, format=fmt, rotation="50 MB", retention=3)


def get_logger(name: str):
    return logger.bind(name=name)
