"""Section classification utilities."""
from __future__ import annotations

import asyncio

from config.PaperQGPrompt import SECTION_CLASSFICATION_PROMPT
from models.llm import BaseChatModel, ChatMessage, LLMError
from services.paper_parser import PaperSection
from utils.logger import get_logger

LOGGER = get_logger(__name__)

PREDEFINED_SECTIONS = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Approach/Methodology/Model/Method",
    "Analysis Theory",
    "Experiments",
    "Experiment Analysis",
    "Discussion/Limitations",
    "Conclusion",
]


class SectionClassifier:
    def __init__(self, chat_model: BaseChatModel, retry_limit: int = 3) -> None:
        self.chat_model = chat_model
        self.retry_limit = retry_limit

    async def classify(self, section: PaperSection) -> str:
        messages = [
            ChatMessage(role="system", content=SECTION_CLASSFICATION_PROMPT),
            ChatMessage(
                role="user",
                content=f"Section title: {section.heading}\nSection content: {section.text}",
            ),
        ]
        attempt = 0
        while attempt <= self.retry_limit:
            try:
                response = await self.chat_model.chat(messages)
                cleaned = response.strip().split("\n")[0]
                if cleaned not in PREDEFINED_SECTIONS:
                    LOGGER.debug("LLM classified section as '%s'", cleaned)
                if cleaned in PREDEFINED_SECTIONS:
                    return cleaned
                return "Other"
            except LLMError as exc:
                attempt += 1
                LOGGER.warning("Section classification failed (attempt %s): %s", attempt, exc)
                await asyncio.sleep(0.5 * attempt)
        return "Other"
