"""Helpers to render evaluation results for the Gradio interface."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from i18n import translate


def build_section_html(evaluations: List[Dict[str, Any]], language: str) -> str:
    parts = ["<div class='section-results'>"]
    for item in evaluations:
        scores = item.get("Scores", {})
        parts.append("<div class='section-item'>")
        parts.append(f"<h3>{translate('section_header', language)}: {item.get('Section', '')}</h3>")
        parts.append(
            f"<p><strong>{translate('section_question', language)}:</strong><br>{item.get('Answer', '').replace('\n', '<br>')}</p>"
        )
        parts.append(f"<h4>{translate('section_scores', language)}</h4>")
        parts.append("<ul>")
        for dimension, details in scores.items():
            parts.append(
                "<li><strong>{}</strong>: {} {:.2f} | {} {:.2f}<br>{}: {}</li>".format(
                    dimension,
                    translate("score", language),
                    float(details.get("Score", 0.0)),
                    translate("confidence", language),
                    float(details.get("Confidence", 0.0)),
                    translate("reason", language),
                    details.get("Reason", ""),
                )
            )
        parts.append("</ul>")
        parts.append("</div>")
    parts.append("</div>")
    return "\n".join(parts)


def build_summary_markdown(result: Dict[str, Any], language: str) -> str:
    lines = ["### {}: {:.2f}".format(translate("final_score", language), result.get("final_score", 0.0))]
    meta = []
    if result.get("title"):
        meta.append(f"**Title:** {result['title']}")
    if result.get("authors"):
        meta.append(f"**Authors:** {result['authors']}")
    if result.get("emails"):
        meta.append(f"**Emails:** {result['emails']}")
    if meta:
        lines.append("\n".join(meta))
    return "\n\n".join(lines)


def result_to_json(result: Dict[str, Any]) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False)
