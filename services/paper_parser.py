"""Parsing utilities for converting MinerU output into structured data."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from models.llm import BaseChatModel, ChatMessage, LLMError
from utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PaperSection:
    heading: str
    text: str


@dataclass
class PaperDocument:
    title: str = ""
    authors: str = ""
    emails: str = ""
    sections: List[PaperSection] = field(default_factory=list)
    references: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "PaperDocument":
        sections = [
            PaperSection(
                heading=str(section.get("heading", "")),
                text=str(section.get("text", "")),
            )
            for section in payload.get("sections", [])
            if section
        ]
        references = []
        raw_refs = payload.get("references", [])
        if isinstance(raw_refs, list):
            references = [str(ref) for ref in raw_refs]
        elif isinstance(raw_refs, str):
            references = [raw_refs]
        return cls(
            title=str(payload.get("title", "")),
            authors=str(payload.get("authors", "")),
            emails=str(payload.get("emails", "")),
            sections=sections,
            references=references,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "emails": self.emails,
            "sections": [section.__dict__ for section in self.sections],
            "references": self.references,
        }


class EnhancedPaperParser:
    """Use an LLM to enhance the parsed structure when requested."""

    PARSE_PROMPT = (
        "You are an expert assistant that normalises academic paper structures.\n"
        "Given the raw JSON extracted from a PDF, return a cleaned JSON with the following schema:\n"
        "{\n"
        "  \"title\": str,\n"
        "  \"authors\": str,\n"
        "  \"emails\": str,\n"
        "  \"sections\": [ { \"heading\": str, \"text\": str } ],\n"
        "  \"references\": [str]\n"
        "}.\n"
        "If any field cannot be determined, leave it as an empty string.\n"
        "Do not include markdown or commentary."
    )

    def __init__(self, chat_model: BaseChatModel) -> None:
        self.chat_model = chat_model

    async def enhance(self, document: PaperDocument) -> PaperDocument:
        payload = json.dumps(document.to_dict(), ensure_ascii=False)
        messages = [
            ChatMessage(role="system", content=self.PARSE_PROMPT),
            ChatMessage(role="user", content=payload),
        ]
        try:
            response = await self.chat_model.chat(messages)
            cleaned = json.loads(response)
        except (json.JSONDecodeError, LLMError) as exc:
            LOGGER.warning("Enhanced parsing failed: %s", exc)
            return document
        try:
            return PaperDocument.from_json(cleaned)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Enhanced parsing produced invalid structure: %s", exc)
            return document
