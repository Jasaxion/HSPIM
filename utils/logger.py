"""Logging utilities for the HSPIM application."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

_STREAM_HANDLER: Optional[logging.Handler] = None
_BASE_FILE_HANDLER: Optional[logging.Handler] = None
_RUN_FILE_HANDLER: Optional[logging.Handler] = None
_CURRENT_LEVEL = logging.INFO
_LOG_DIR = Path("logs")


def _ensure_stream_handler() -> None:
    """Attach a single stream handler to the root logger."""

    global _STREAM_HANDLER
    if _STREAM_HANDLER is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(handler)
        _STREAM_HANDLER = handler


def _replace_handler(
    attribute: str, handler: Optional[logging.Handler], new_handler: Optional[logging.Handler]
) -> Optional[logging.Handler]:
    """Swap a handler on the root logger and return the active handler."""

    root_logger = logging.getLogger()
    if handler is not None:
        if handler in root_logger.handlers:
            root_logger.removeHandler(handler)
        handler.close()
    if new_handler is not None:
        root_logger.addHandler(new_handler)
    globals()[attribute] = new_handler
    return new_handler


def _set_level(level: int) -> None:
    global _CURRENT_LEVEL
    _CURRENT_LEVEL = level
    root = logging.getLogger()
    root.setLevel(level)
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.setLevel(level)


def configure_logging(
    settings: Optional[Dict[str, Any]] = None,
    base_log_file: Optional[Path] = None,
) -> Path:
    """Configure root logging handlers based on settings and return the log path."""

    global _LOG_DIR, _BASE_FILE_HANDLER

    enable_logging = True
    debug_mode = False
    logs_dir = _LOG_DIR

    if settings:
        logs_dir = Path(settings.get("logs_dir") or logs_dir)
        enable_logging = bool(settings.get("enable_logging", True))
        debug_mode = bool(settings.get("debug_mode", False))

    logs_dir.mkdir(parents=True, exist_ok=True)
    _LOG_DIR = logs_dir

    level = logging.DEBUG if debug_mode else logging.INFO
    _ensure_stream_handler()
    _set_level(level)

    if not enable_logging:
        _replace_handler("_BASE_FILE_HANDLER", _BASE_FILE_HANDLER, None)
        return logs_dir

    log_path = base_log_file or logs_dir / ("debug.log" if debug_mode else "app.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    _replace_handler("_BASE_FILE_HANDLER", _BASE_FILE_HANDLER, file_handler)
    return log_path


def activate_run_log(log_file: Path) -> None:
    """Attach a per-run log file handler."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    _replace_handler("_RUN_FILE_HANDLER", _RUN_FILE_HANDLER, handler)


def deactivate_run_log() -> None:
    """Remove the per-run log file handler if present."""

    _replace_handler("_RUN_FILE_HANDLER", _RUN_FILE_HANDLER, None)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a logger configured to use shared handlers."""

    logger = logging.getLogger(name)
    if level is None:
        level = _CURRENT_LEVEL
    logger.setLevel(level)
    logger.propagate = True
    _ensure_stream_handler()
    return logger


def set_global_level(level: int) -> None:
    """Manually override the logging level for all loggers."""

    _set_level(level)
