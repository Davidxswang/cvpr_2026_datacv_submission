"""Logging setup: file (DEBUG) + terminal (WARNING)."""

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(exp_dir: Path) -> None:
    """Configure root logger with dual handlers.

    - File handler  → exp_dir/run.log at DEBUG level (captures everything)
    - Console handler → WARNING level (only important messages in terminal)
    """
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "run.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates on re-init
    root.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # File handler: DEBUG level
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Console handler: WARNING level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    logging.info("Logging initialized → %s", log_path)
