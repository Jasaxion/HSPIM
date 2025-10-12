"""Utilities for loading and persisting application configuration."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

from . import ModelConfig

CONFIG_FILE = Path(ModelConfig.__file__)


class ConfigError(Exception):
    """Raised when configuration values are invalid."""


def _load_module_dict() -> Dict[str, Any]:
    """Dynamically import the ModelConfig module to obtain CONFIG."""
    spec = importlib.util.spec_from_file_location("ModelConfig", CONFIG_FILE)
    if spec is None or spec.loader is None:
        raise ConfigError("Unable to load configuration module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("ModelConfig_runtime", None)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)  # type: ignore[call-arg]
    if not hasattr(module, "CONFIG"):
        raise ConfigError("CONFIG dictionary missing from ModelConfig.py")
    config = getattr(module, "CONFIG")
    if not isinstance(config, dict):
        raise ConfigError("CONFIG must be a dictionary.")
    return config


def load_config() -> Dict[str, Any]:
    """Return a deep copy of the configuration dictionary."""
    config = _load_module_dict()
    return json.loads(json.dumps(config))  # simple deep copy


def _ensure_directories(config: Dict[str, Any]) -> None:
    general = config.get("general", {})
    for key in ("data_dir", "review_dir", "output_dir"):
        path = general.get(key)
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)


def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if "model" not in config:
        raise ConfigError("Missing 'model' configuration block.")
    model_cfg = config["model"]
    provider = model_cfg.get("provider")
    if provider not in ("OPENAI", "TRANSFORMERS"):
        raise ConfigError("Unsupported model provider: %s" % provider)
    max_workers = config.get("general", {}).get("max_workers", 1)
    if not isinstance(max_workers, int) or max_workers < 1:
        raise ConfigError("'max_workers' must be a positive integer.")
    retry_limit = config.get("general", {}).get("retry_limit", 1)
    if not isinstance(retry_limit, int) or retry_limit < 0:
        raise ConfigError("'retry_limit' must be a non-negative integer.")
    return config


def save_config(config: Dict[str, Any]) -> None:
    """Persist the configuration back to ModelConfig.py."""
    validated = _validate_config(config)
    _ensure_directories(validated)
    formatted = json.dumps(validated, indent=4, ensure_ascii=False, sort_keys=True)
    content = (
        "\"\"\"Model configuration for the HSPIM application.\n\n"
        "This file is auto-generated and will be overwritten when configuration\n"
        "settings are updated from the Gradio interface. Avoid manual edits unless\n"
        "necessary. All values are stored as standard Python literals to simplify\n"
        "parsing and persistence.\n\"\"\"\n"
        "from __future__ import annotations\n\n"
        "from typing import Any, Dict\n\n"
        "CONFIG: Dict[str, Any] = "
        f"{formatted}\n"
    )
    CONFIG_FILE.write_text(content + "\n", encoding="utf-8")


__all__ = ["load_config", "save_config", "ConfigError"]
