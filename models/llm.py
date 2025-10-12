"""LLM client abstractions supporting both API and local models."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ChatMessage:
    role: str
    content: str


class LLMError(Exception):
    """Raised when LLM generation fails."""


class BaseChatModel:
    """Base class for chat completion style models."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def chat(self, messages: Iterable[ChatMessage], **options: Any) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._chat_sync, list(messages), options)

    # pylint: disable=unused-argument
    def _chat_sync(self, messages: List[ChatMessage], options: Dict[str, Any]) -> str:
        raise NotImplementedError


class OpenAIChatModel(BaseChatModel):
    """Chat model backed by OpenAI compatible APIs."""

    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.2,
                 max_tokens: int = 2048, request_timeout: int = 120, **_: Any) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise LLMError("openai package is required for OpenAI provider") from exc
        if not api_key:
            raise LLMError("API key is required for OpenAI provider")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout

    def _chat_sync(self, messages: List[ChatMessage], options: Dict[str, Any]) -> str:
        payload = [message.__dict__ for message in messages]
        try:
            response = self.client.responses.create(
                model=self.model,
                input=payload,
                temperature=options.get("temperature", self.temperature),
                max_output_tokens=options.get("max_tokens", self.max_tokens),
                timeout=options.get("request_timeout", self.request_timeout),
            )
        except Exception as exc:  # pragma: no cover
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        try:
            output = response.output[0].content[0].text  # type: ignore[attr-defined]
        except (AttributeError, IndexError, KeyError) as exc:  # pragma: no cover
            raise LLMError("Unexpected OpenAI response format") from exc
        return str(output)


def _format_prompt(messages: List[ChatMessage], tokenizer: Optional[Any]) -> str:
    """Render chat messages using a tokenizer chat template when available."""

    payload = [message.__dict__ for message in messages]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(  # type: ignore[call-arg]
                payload,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to apply chat template: %s", exc)
    return "\n".join(f"{msg.role.upper()}: {msg.content}" for msg in messages)


def _resolve_torch_dtype(dtype: str) -> Optional[Any]:
    if not dtype or dtype.lower() == "auto":
        return None
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        LOGGER.warning("torch is required to set dtype '%s': %s", dtype, exc)
        return None
    attr = getattr(torch, dtype, None)
    if attr is None:
        LOGGER.warning("Unsupported torch dtype '%s'", dtype)
    return attr


class TransformersChatModel(BaseChatModel):
    """Local model backed by HuggingFace transformers."""

    def __init__(
        self,
        model: str = "",
        local_path: str = "",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 512,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
        chat_template: str = "",
        **_: Any,
    ) -> None:
        model_id = local_path or model
        if not model_id:
            raise LLMError("Transformers provider requires a model path or identifier.")
        super().__init__(model=model_id)
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "transformers package is required for local models"
            ) from exc
        kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        resolved_dtype = _resolve_torch_dtype(dtype)
        if resolved_dtype is not None:
            kwargs["torch_dtype"] = resolved_dtype
        if pipeline_kwargs:
            kwargs.update(pipeline_kwargs)
        self.generator = pipeline("text-generation", model=model_id, **kwargs)
        self.tokenizer = getattr(self.generator, "tokenizer", None)
        if chat_template and self.tokenizer is not None:
            try:
                self.tokenizer.chat_template = chat_template  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover - tokenizer may not allow override
                LOGGER.warning("Tokenizer does not support chat_template override.")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _chat_sync(self, messages: List[ChatMessage], options: Dict[str, Any]) -> str:
        prompt = _format_prompt(messages, self.tokenizer)
        result = self.generator(
            prompt,
            max_new_tokens=options.get("max_tokens", self.max_tokens),
            temperature=options.get("temperature", self.temperature),
            do_sample=True,
        )
        try:
            return result[0]["generated_text"]
        except (KeyError, IndexError) as exc:  # pragma: no cover
            raise LLMError("Unexpected transformers output format") from exc


class VLLMChatModel(BaseChatModel):
    """Local inference powered by vLLM."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.7,
        max_tokens: int = 512,
        chat_template: str = "",
        engine_kwargs: Optional[Dict[str, Any]] = None,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        if not model_path:
            raise LLMError("vLLM engine requires a model path or identifier.")
        super().__init__(model=model_path)
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise LLMError("vllm package is required for engine 'vllm'") from exc
        llm_kwargs: Dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": trust_remote_code,
        }
        if dtype and dtype.lower() != "auto":
            llm_kwargs["dtype"] = dtype
        if gpu_memory_utilization:
            llm_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)
        if engine_kwargs:
            llm_kwargs.update(engine_kwargs)
        self.llm = LLM(**llm_kwargs)
        self._sampling_class = SamplingParams
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_sampling_kwargs = sampling_kwargs or {}
        self.tokenizer = self.llm.get_tokenizer()
        if chat_template and self.tokenizer is not None:
            try:
                self.tokenizer.chat_template = chat_template  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover
                LOGGER.warning("Tokenizer does not support chat_template override.")

    def _chat_sync(self, messages: List[ChatMessage], options: Dict[str, Any]) -> str:
        prompt = _format_prompt(messages, self.tokenizer)
        sampling_options = dict(self.default_sampling_kwargs)
        sampling_options["temperature"] = options.get("temperature", self.temperature)
        sampling_options["max_tokens"] = options.get("max_tokens", self.max_tokens)
        sampling = self._sampling_class(**sampling_options)
        outputs = self.llm.generate([prompt], sampling_params=sampling)
        try:
            return outputs[0].outputs[0].text  # type: ignore[attr-defined]
        except (AttributeError, IndexError) as exc:  # pragma: no cover
            raise LLMError("Unexpected vLLM output format") from exc


def build_chat_model(config: Dict[str, Any]) -> BaseChatModel:
    provider = config.get("provider", "OPENAI").upper()
    if provider == "OPENAI":
        return OpenAIChatModel(
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url", "https://api.openai.com/v1"),
            model=config.get("name", "gpt-4o-mini"),
            temperature=config.get("temperature", 0.2),
            max_tokens=config.get("max_tokens", 2048),
            request_timeout=config.get("request_timeout", 120),
        )
    if provider == "TRANSFORMERS":
        engine = (config.get("engine") or "transformers").lower()
        extra = config.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}
        if engine == "vllm":
            return VLLMChatModel(
                model_path=config.get("local_path") or config.get("name", ""),
                tensor_parallel_size=int(config.get("tensor_parallel_size", 1)),
                dtype=config.get("dtype", "auto"),
                trust_remote_code=bool(config.get("trust_remote_code", True)),
                gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.9)),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 512),
                chat_template=config.get("chat_template", ""),
                engine_kwargs=extra.get("engine_kwargs") if isinstance(extra.get("engine_kwargs"), dict) else extra,
                sampling_kwargs=extra.get("sampling_params") if isinstance(extra.get("sampling_params"), dict) else None,
            )
        pipeline_kwargs = extra.get("pipeline_kwargs") if isinstance(extra.get("pipeline_kwargs"), dict) else extra
        return TransformersChatModel(
            model=config.get("name", ""),
            local_path=config.get("local_path", ""),
            dtype=config.get("dtype", "auto"),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 512),
            pipeline_kwargs=pipeline_kwargs if isinstance(pipeline_kwargs, dict) else {},
            chat_template=config.get("chat_template", ""),
        )
    raise LLMError(f"Unsupported provider: {provider}")


def messages_from_dict(payload: Iterable[Dict[str, str]]) -> List[ChatMessage]:
    return [ChatMessage(role=item["role"], content=item["content"]) for item in payload]


__all__ = [
    "ChatMessage",
    "BaseChatModel",
    "OpenAIChatModel",
    "TransformersChatModel",
    "VLLMChatModel",
    "build_chat_model",
    "LLMError",
    "messages_from_dict",
]
