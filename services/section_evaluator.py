"""Section evaluation logic following the HSPIM pipeline."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from config.PaperQGPrompt import (
    NOVELTY_QA_PROMPT,
    QG_BASE_PROMPT,
    QG_PREDEFINED_QUESTION_COMMON,
    QG_PREDEFINED_QUESTION_SPECIFIC,
)
from models.llm import BaseChatModel, ChatMessage, LLMError
from services.paper_parser import PaperDocument, PaperSection
from utils.logger import get_logger

LOGGER = get_logger(__name__)

DIMENSIONS = ["Novelty", "Contribution", "Feasibility"]


def calculate_paper_predicted_score(evaluations: List[Dict[str, Any]]) -> float:
    evaluation_scores = []
    for eval_item in evaluations:
        scores_data = eval_item.get("Scores", {})
        total_weighted_score = 0.0
        total_confidence = 0.0
        for dim in DIMENSIONS:
            dim_data = scores_data.get(dim)
            if isinstance(dim_data, dict):
                try:
                    score = float(dim_data.get("Score", 0.0))
                    confidence = float(dim_data.get("Confidence", 0.0))
                except (ValueError, TypeError):
                    continue
                total_weighted_score += score * confidence
                total_confidence += confidence
        if total_confidence > 0:
            evaluation_scores.append(total_weighted_score / total_confidence)
    if not evaluation_scores:
        return 0.0
    return sum(evaluation_scores) / len(evaluation_scores)


class SectionEvaluator:
    def __init__(
        self,
        chat_model: BaseChatModel,
        max_workers: int = 4,
        retry_limit: int = 3,
    ) -> None:
        self.chat_model = chat_model
        self.max_workers = max_workers
        self.retry_limit = retry_limit
        from services.section_classifier import SectionClassifier

        self.classifier = SectionClassifier(chat_model, retry_limit=retry_limit)

    async def evaluate_document(self, document: PaperDocument) -> Dict[str, Any]:
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_section(section: PaperSection) -> Dict[str, Any]:
            async with semaphore:
                matched = await self._classify_section(section)
                answer = await self._generate_answer(section, matched)
                scores = await self._score_section(section, matched, answer)
                return {
                    "Section": section.heading,
                    "MatchedSection": matched,
                    "Answer": answer,
                    "Scores": scores,
                }

        tasks = [process_section(section) for section in document.sections]
        results = await asyncio.gather(*tasks)
        final_score = calculate_paper_predicted_score(results)
        return {
            "title": document.title,
            "authors": document.authors,
            "emails": document.emails,
            "evaluations": results,
            "final_score": final_score,
        }

    async def _classify_section(self, section: PaperSection) -> str:
        return await self.classifier.classify(section)

    async def _generate_answer(self, section: PaperSection, matched: str) -> str:
        question = QG_PREDEFINED_QUESTION_SPECIFIC.get(matched, "")
        question = f"{question} {QG_PREDEFINED_QUESTION_COMMON}".strip()
        messages = [
            ChatMessage(role="system", content=f"{QG_BASE_PROMPT} {question}"),
            ChatMessage(
                role="user",
                content=(
                    f"Given the section title: {section.heading}\n"
                    f"Content: {section.text}\n"
                    "Please provide a direct response to the question."
                ),
            ),
        ]
        return await self._retry_chat(messages)

    async def _score_section(self, section: PaperSection, matched: str, answer: str) -> Dict[str, Any]:
        question = QG_PREDEFINED_QUESTION_SPECIFIC.get(matched, "")
        question = f"{question} {QG_PREDEFINED_QUESTION_COMMON}".strip()
        prompt = (
            f"Please output JSON OUTPUT{{Novelty, Contribution, Feasibility}}. "
            f"Given the Section Title: {section.heading}\n"
            f"Content: {section.text}\n"
            f"Question: {question}\n"
            f"Answer: {answer}"
        )
        messages = [
            ChatMessage(role="system", content=NOVELTY_QA_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]
        raw = await self._retry_chat(messages)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse JSON scores for section '%s'", section.heading)
            parsed = {}
        scores: Dict[str, Any] = {}
        for dim in DIMENSIONS:
            entry = parsed.get(dim, {}) if isinstance(parsed, dict) else {}
            if isinstance(entry, dict):
                scores[dim] = {
                    "Score": entry.get("Score", 0.0),
                    "Reason": entry.get("Reason", ""),
                    "Confidence": entry.get("Confidence", 0.0),
                }
            else:
                scores[dim] = {"Score": 0.0, "Reason": "", "Confidence": 0.0}
        return scores

    async def _retry_chat(self, messages: List[ChatMessage]) -> str:
        attempt = 0
        delay = 0.5
        while attempt <= self.retry_limit:
            try:
                return await self.chat_model.chat(messages)
            except LLMError as exc:
                attempt += 1
                LOGGER.warning("LLM call failed (attempt %s/%s): %s", attempt, self.retry_limit, exc)
                await asyncio.sleep(delay)
                delay *= 2
        raise LLMError("LLM request failed after retries")
