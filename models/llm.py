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


class TransformersChatModel(BaseChatModel):
    """Local model backed by HuggingFace transformers."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "transformers package is required for local models"
            ) from exc
        self.generator = pipeline("text-generation", model=model, trust_remote_code=True)

    def _chat_sync(self, messages: List[ChatMessage], options: Dict[str, Any]) -> str:
        prompt = "\n".join(f"{msg.role.upper()}: {msg.content}" for msg in messages)
        result = self.generator(
            prompt,
            max_new_tokens=options.get("max_tokens", 512),
            temperature=options.get("temperature", 0.7),
            do_sample=True,
        )
        try:
            return result[0]["generated_text"]
        except (KeyError, IndexError) as exc:  # pragma: no cover
            raise LLMError("Unexpected transformers output format") from exc


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
        return TransformersChatModel(model=config.get("name", "gpt2"))
    raise LLMError(f"Unsupported provider: {provider}")


def messages_from_dict(payload: Iterable[Dict[str, str]]) -> List[ChatMessage]:
    return [ChatMessage(role=item["role"], content=item["content"]) for item in payload]


__all__ = [
    "ChatMessage",
    "BaseChatModel",
    "OpenAIChatModel",
    "TransformersChatModel",
    "build_chat_model",
    "LLMError",
    "messages_from_dict",
]
