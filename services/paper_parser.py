"""Parsing utilities for converting MinerU output into structured data."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from models.llm import BaseChatModel, ChatMessage, LLMError
from utils.logger import get_logger

LOGGER = get_logger(__name__)


_ABSTRACT_PATTERN = re.compile(r"^(abstract|summary|摘[要]|概要)\b[:：\s\-–—]*", re.IGNORECASE)
_INDEX_TERMS_PATTERN = re.compile(r"^(index\s+terms?|keywords?)\b", re.IGNORECASE)


def _extract_references(text: str) -> List[str]:
    references: List[str] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^[\-•\d\.)\s]+", "", cleaned)
        if cleaned:
            references.append(cleaned)
    return references


def _process_metadata(lines: List[str]) -> Tuple[str, str, str]:
    """Extract authors, emails, and abstract text from metadata lines."""

    authors_parts: List[str] = []
    emails_parts: List[str] = []
    abstract_lines: List[str] = []
    abstract_mode = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if abstract_mode:
            if _INDEX_TERMS_PATTERN.match(line):
                abstract_mode = False
            else:
                abstract_lines.append(line)
                continue

        abstract_match = _ABSTRACT_PATTERN.match(line)
        if abstract_match:
            cleaned = line[abstract_match.end():].strip()
            abstract_lines.append(cleaned or line)
            abstract_mode = True
            continue

        if "@" in line:
            emails_parts.append(line)
            continue

        authors_parts.append(line)

    authors = " ".join(authors_parts).strip()
    emails = " ".join(emails_parts).strip()
    abstract = "\n".join(abstract_lines).strip()
    return authors, emails, abstract


def _ensure_abstract_section(
    sections: List[PaperSection], abstract_text: str
) -> None:
    if not abstract_text:
        return
    for section in sections:
        if "abstract" in section.heading.lower():
            return
    sections.insert(0, PaperSection("Abstract", abstract_text))


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
        references: List[str] = []
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

    @classmethod
    def from_markdown(cls, markdown: str) -> "PaperDocument":
        lines = markdown.splitlines()
        title = ""
        authors = ""
        emails = ""
        sections: List[PaperSection] = []
        references: List[str] = []

        current_heading: Optional[str] = None
        current_lines: List[str] = []
        metadata_buffer: List[str] = []

        heading_pattern = re.compile(r"^(#+)\s+(.*)")

        for raw_line in lines:
            line = raw_line.rstrip()
            match = heading_pattern.match(line.strip())
            if match:
                level = len(match.group(1))
                heading_text = match.group(2).strip()
                if level <= 2 and not title:
                    title = heading_text
                    current_heading = None
                    current_lines = []
                    continue
                if current_heading:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        if "reference" in current_heading.lower():
                            references = _extract_references(section_text)
                        else:
                            sections.append(PaperSection(current_heading, section_text))
                current_heading = heading_text
                current_lines = []
                continue

            if current_heading:
                current_lines.append(line)
            else:
                if line.strip():
                    metadata_buffer.append(line.strip())

        if current_heading:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                if "reference" in current_heading.lower():
                    references = _extract_references(section_text)
                else:
                    sections.append(PaperSection(current_heading, section_text))

        if metadata_buffer:
            authors, emails, abstract_text = _process_metadata(metadata_buffer)
            if abstract_text:
                _ensure_abstract_section(sections, abstract_text)

        if not sections:
            fallback_text = markdown.strip()
            if fallback_text:
                sections = [PaperSection("Content", fallback_text)]

        return cls(title=title, authors=authors, emails=emails, sections=sections, references=references)

    @classmethod
    def from_segments(cls, segments: Iterable[Dict[str, Any]]) -> "PaperDocument":
        segment_list = [entry for entry in segments if isinstance(entry, dict)]
        title = ""
        authors = ""
        emails = ""
        sections: List[PaperSection] = []
        references: List[str] = []

        current_heading: Optional[str] = None
        current_lines: List[str] = []
        metadata_buffer: List[str] = []

        for entry in segment_list:
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            text_level = entry.get("text_level")
            if isinstance(text_level, int) and text_level <= 2:
                if not title:
                    title = text
                    continue
                if current_heading:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        if "reference" in current_heading.lower():
                            references = _extract_references(section_text)
                        else:
                            sections.append(PaperSection(current_heading, section_text))
                current_heading = text
                current_lines = []
                continue

            if current_heading:
                current_lines.append(text)
            else:
                metadata_buffer.append(text)

        if current_heading:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                if "reference" in current_heading.lower():
                    references = _extract_references(section_text)
                else:
                    sections.append(PaperSection(current_heading, section_text))

        if metadata_buffer:
            authors, emails, abstract_text = _process_metadata(metadata_buffer)
            if abstract_text:
                _ensure_abstract_section(sections, abstract_text)

        if not sections and segment_list:
            combined_text = "\n".join(
                str(entry.get("text", "")) for entry in segment_list
            ).strip()
            if combined_text:
                sections = [PaperSection("Content", combined_text)]

        return cls(title=title, authors=authors, emails=emails, sections=sections, references=references)

    @classmethod
    def from_mineru(
        cls, payload: Any, markdown: Optional[str] = None
    ) -> "PaperDocument":
        if isinstance(payload, dict) and "sections" in payload:
            return cls.from_json(payload)
        if markdown:
            try:
                return cls.from_markdown(markdown)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to parse MinerU markdown: %s", exc)
        if isinstance(payload, list):
            return cls.from_segments(payload)
        if isinstance(payload, dict):
            return cls.from_segments(payload.values())
        LOGGER.warning("Unsupported MinerU payload type: %s", type(payload))
        return cls()

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
