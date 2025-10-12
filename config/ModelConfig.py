"""Model configuration for the HSPIM application.

This file is auto-generated and will be overwritten when configuration
settings are updated from the Gradio interface. Avoid manual edits unless
necessary. All values are stored as standard Python literals to simplify
parsing and persistence.
"""
from __future__ import annotations

from typing import Any, Dict

# The CONFIG dictionary stores runtime parameters for external services,
# model providers, and application level behaviours. The structure is
# intentionally flat so it can be serialised back into this module without
# complex templating logic.
CONFIG: Dict[str, Any] = {
    "model": {
        "provider": "OPENAI",
        "name": "gpt-4o-mini",
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "local_path": "",
        "engine": "transformers",
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.9,
        "chat_template": "",
        "temperature": 0.2,
        "max_tokens": 2048,
        "request_timeout": 120,
        "extra": {}
    },
    "general": {
        "data_dir": "data",
        "review_dir": "data",
        "output_dir": "data",
        "overwrite_existing_files": False,
        "max_workers": 16,
        "retry_limit": 3,
        "enable_logging": True
    },
    "mineru": {
        "api_key": "",
        "base_url": "https://mineru.net/api/v4",
        "use_ocr": True,
        "enable_formula": False,
        "enable_table": True,
        "language": "en"
    },
    "available_models": {
        "OPENAI": {
            "description": "OpenAI compatible APIs including Azure/Deepseek endpoints.",
            "fields": ["api_key", "base_url", "name"]
        },
        "TRANSFORMERS": {
            "description": "Local HuggingFace or vLLM-hosted models.",
            "fields": [
                "local_path",
                "engine",
                "tensor_parallel_size",
                "dtype",
                "trust_remote_code",
                "gpu_memory_utilization",
                "chat_template"
            ]
        }
    },
    "ui": {
        "default_language": "en",
        "theme": "soft",
        "enhanced_parsing_default": False
    }
}
