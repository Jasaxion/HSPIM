"""High level orchestration for the innovation analysis pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from config.config_manager import load_config
from models.llm import LLMError, build_chat_model
from services.mineru import MineruClient, MineruError
from services.paper_parser import EnhancedPaperParser, PaperDocument
from services.section_evaluator import SectionEvaluator
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class PipelineError(Exception):
    pass


async def analyse_paper(file_path: Path, use_enhanced: bool = False) -> Dict[str, object]:
    config = load_config()
    general_cfg = config.get("general", {})
    model_cfg = config.get("model", {})
    mineru_cfg = config.get("mineru", {})

    try:
        chat_model = build_chat_model(model_cfg)
    except LLMError as exc:
        raise PipelineError(f"Failed to initialise model: {exc}") from exc

    mineru_client = MineruClient(
        api_key=mineru_cfg.get("api_key", ""),
        base_url=mineru_cfg.get("base_url", "https://mineru.net/api/v4"),
    )

    try:
        raw_json = mineru_client.parse(
            file_path,
            is_ocr=mineru_cfg.get("use_ocr", True),
            enable_formula=mineru_cfg.get("enable_formula", False),
            enable_table=mineru_cfg.get("enable_table", True),
            language=mineru_cfg.get("language", "en"),
        )
    except MineruError as exc:
        raise PipelineError(str(exc)) from exc

    document = PaperDocument.from_json(raw_json)

    if use_enhanced:
        enhancer = EnhancedPaperParser(chat_model)
        document = await enhancer.enhance(document)

    evaluator = SectionEvaluator(
        chat_model,
        max_workers=general_cfg.get("max_workers", 4),
        retry_limit=general_cfg.get("retry_limit", 3),
    )
    return await evaluator.evaluate_document(document)
