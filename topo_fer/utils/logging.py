"""Logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
     """Create or retrieve a logger with a consistent format."""
     logger = logging.getLogger(name)

     if not logger.handlers:
         handler = logging.StreamHandler()
         formatter = logging.Formatter(
             "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
             datefmt="%Y-%m-%d %H:%M:%S",
         )
         handler.setFormatter(formatter)
         logger.addHandler(handler)
         logger.setLevel(logging.INFO)

     return logger


def configure_file_logger(logger: logging.Logger, log_dir: str, filename: str) -> None:
     """Attach a rotating file handler to the logger."""
     Path(log_dir).mkdir(parents=True, exist_ok=True)
     file_handler = logging.FileHandler(Path(log_dir) / filename)
     formatter = logging.Formatter(
         "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
         datefmt="%Y-%m-%d %H:%M:%S",
     )
     file_handler.setFormatter(formatter)
     logger.addHandler(file_handler)
