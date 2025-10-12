"""High level orchestration for the innovation analysis pipeline."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from config.config_manager import load_config
from models.llm import LLMError, build_chat_model
from services.mineru import MineruClient, MineruError
from services.paper_parser import EnhancedPaperParser, PaperDocument
from services.section_evaluator import SectionEvaluator
from utils.logger import (
    activate_run_log,
    configure_logging,
    deactivate_run_log,
    get_logger,
)

LOGGER = get_logger(__name__)


class PipelineError(Exception):
    pass


def _ensure_run_directory(general_cfg: Dict[str, Any], file_path: Path) -> Path:
    logs_root = Path(general_cfg.get("logs_dir") or "logs")
    logs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    stem = file_path.stem or "paper"
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    safe_stem = safe_stem[:40] or "paper"
    run_dir = logs_root / f"{timestamp}_{safe_stem}_{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: Any) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to write JSON debug artefact %s: %s", path, exc)


async def analyse_paper(file_path: Path, use_enhanced: bool = False) -> Dict[str, object]:
    config = load_config()
    general_cfg = config.get("general", {})
    model_cfg = config.get("model", {})
    mineru_cfg = config.get("mineru", {})

    configure_logging(general_cfg)
    run_dir = _ensure_run_directory(general_cfg, file_path)
    activate_run_log(run_dir / "pipeline.log")
    LOGGER.info(
        "Starting analysis for %s (enhanced=%s)", file_path.name, "yes" if use_enhanced else "no"
    )

    try:
        try:
            chat_model = build_chat_model(model_cfg)
        except LLMError as exc:
            LOGGER.exception("Failed to initialise model")
            raise PipelineError(f"Failed to initialise model: {exc}") from exc

        mineru_client = MineruClient(
            api_key=mineru_cfg.get("api_key", ""),
            base_url=mineru_cfg.get("base_url", "https://mineru.net/api/v4"),
        )

        try:
            mineru_output = mineru_client.parse(
                file_path,
                is_ocr=mineru_cfg.get("use_ocr", True),
                enable_formula=mineru_cfg.get("enable_formula", False),
                enable_table=mineru_cfg.get("enable_table", True),
                language=mineru_cfg.get("language", "en"),
            )
        except MineruError as exc:
            LOGGER.exception("MinerU parsing failed")
            raise PipelineError(str(exc)) from exc

        raw_payload = mineru_output.get("raw") if isinstance(mineru_output, dict) else None
        markdown_payload = (
            mineru_output.get("markdown") if isinstance(mineru_output, dict) else None
        )
        if raw_payload is None:
            raise PipelineError("MinerU did not return a JSON payload")

        mineru_json_path = run_dir / "mineru_raw.json"
        _write_json(mineru_json_path, raw_payload)
        LOGGER.info("MinerU JSON stored at %s", mineru_json_path)
        if markdown_payload:
            markdown_path = run_dir / "mineru_markdown.md"
            markdown_path.write_text(markdown_payload, encoding="utf-8")
            LOGGER.info("MinerU markdown stored at %s", markdown_path)

        document = PaperDocument.from_mineru(raw_payload, markdown_payload)
        initial_doc_path = run_dir / "document_initial.json"
        _write_json(initial_doc_path, document.to_dict())
        LOGGER.debug("Initial document saved to %s", initial_doc_path)
        LOGGER.debug("Initial document parsed with %d sections", len(document.sections))

        if use_enhanced:
            LOGGER.info("Running enhanced parsing")
            enhancer = EnhancedPaperParser(chat_model)
            document = await enhancer.enhance(document)
            enhanced_doc_path = run_dir / "document_enhanced.json"
            _write_json(enhanced_doc_path, document.to_dict())
            LOGGER.debug("Enhanced document saved to %s", enhanced_doc_path)
            LOGGER.debug("Enhanced document now has %d sections", len(document.sections))

        evaluator = SectionEvaluator(
            chat_model,
            max_workers=general_cfg.get("max_workers", 4),
            retry_limit=general_cfg.get("retry_limit", 3),
        )
        LOGGER.info("Evaluating %d sections", len(document.sections))
        result = await evaluator.evaluate_document(document)
        result_with_logs = dict(result)
        result_with_logs["log_dir"] = str(run_dir.resolve())
        evaluation_path = run_dir / "evaluation.json"
        _write_json(evaluation_path, result_with_logs)
        LOGGER.debug("Evaluation details saved to %s", evaluation_path)
        LOGGER.info("Final predicted innovation score: %.2f", result_with_logs.get("final_score", 0.0))
        return result_with_logs
    finally:
        deactivate_run_log()
