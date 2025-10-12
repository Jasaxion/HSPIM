"""Logging utilities for the HSPIM application."""
from __future__ import annotations

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def set_global_level(level: int) -> None:
    """Set the logging level for all existing loggers."""
    logging.basicConfig(level=level)
    for logger_name in logging.Logger.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(level)
