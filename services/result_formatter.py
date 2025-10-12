"""Helpers to render evaluation results for the Gradio interface."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from i18n import translate


def build_section_html(evaluations: List[Dict[str, Any]], language: str) -> str:
    parts = ["<div class='section-results'>"]
    for item in evaluations:
        scores = item.get("Scores", {})
        section_title = item.get("Section", "")
        answer_text = item.get("Answer", "") or ""
        answer_html = answer_text.replace("\n", "<br>")
        parts.append("<div class='section-item'>")
        parts.append(
            "<h3>{header}: {title}</h3>".format(
                header=translate("section_header", language),
                title=section_title,
            )
        )
        parts.append(
            "<p class='section-answer'><strong>{question}</strong><br>{answer}</p>".format(
                question=translate("section_question", language),
                answer=answer_html,
            )
        )
        parts.append(
            "<h4 class='section-scores-title'>{}</h4>".format(
                translate("section_scores", language)
            )
        )
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
    lines = [
        "### {}: {:.2f}".format(
            translate("final_score", language), result.get("final_score", 0.0)
        )
    ]
    meta = []
    if result.get("title"):
        meta.append(
            "**{}:** {}".format(
                translate("title_label", language), result["title"]
            )
        )
    if result.get("authors"):
        meta.append(
            "**{}:** {}".format(
                translate("authors_label", language), result["authors"]
            )
        )
    if result.get("emails"):
        meta.append(
            "**{}:** {}".format(
                translate("emails_label", language), result["emails"]
            )
        )
    if meta:
        lines.append("\n".join(meta))
    return "\n\n".join(lines)


def result_to_json(result: Dict[str, Any]) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False)
