"""Gradio application for the HSPIM paper innovation analysis."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr

from config.config_manager import ConfigError, load_config, save_config
from i18n import translate
from services.pipeline import PipelineError, analyse_paper
from services.result_formatter import (
    build_section_html,
    build_summary_markdown,
)


async def _run_analysis(file_obj, use_enhanced: bool, language: str) -> Tuple[str, str, Dict[str, object]]:
    if file_obj is None:
        return translate("no_file", language), "", {}
    temp_dir = Path(tempfile.mkdtemp())
    uploaded_path = temp_dir / Path(file_obj.name).name
    shutil.copy(file_obj.name, uploaded_path)
    try:
        result = await analyse_paper(uploaded_path, use_enhanced=use_enhanced)
    except PipelineError as exc:
        return f"**{translate('error_prefix', language)}:** {exc}", "", {}
    summary = build_summary_markdown(result, language)
    sections_html = build_section_html(result.get("evaluations", []), language)
    return summary, sections_html, result


def _load_config_text() -> str:
    config = load_config()
    return json.dumps(config, indent=2, ensure_ascii=False)


def _save_config_from_text(config_text: str, language: str) -> str:
    try:
        payload = json.loads(config_text)
        save_config(payload)
    except (json.JSONDecodeError, ConfigError) as exc:
        return translate("config_error", language).format(error=exc)
    return translate("success_config", language)


def build_app() -> gr.Blocks:
    config = load_config()
    ui_config = config.get("ui", {})
    default_language = ui_config.get("default_language", "en")
    theme = ui_config.get("theme", "soft")
    enhanced_default = ui_config.get("enhanced_parsing_default", False)

    with gr.Blocks(theme=theme, title=translate("app_title", default_language)) as demo:
        gr.Markdown(f"# {translate('app_title', default_language)}")

        with gr.Tab(translate("analysis_tab", default_language)):
            language_input = gr.Radio(
                choices=["en", "zh"],
                value=default_language,
                label=translate("language_label", default_language),
            )
            with gr.Row():
                file_input = gr.File(label=translate("upload_label", default_language))
                enhanced_toggle = gr.Checkbox(
                    value=enhanced_default,
                    label=translate("enhanced_parsing", default_language),
                )
            analyze_button = gr.Button(translate("analyze_button", default_language))
            summary_output = gr.Markdown()
            sections_output = gr.HTML()
            raw_output = gr.JSON(label=translate("raw_output_title", default_language))

            async def on_analyze(file_obj, enhanced, language):
                return await _run_analysis(file_obj, enhanced, language or default_language)

            analyze_button.click(
                fn=on_analyze,
                inputs=[file_input, enhanced_toggle, language_input],
                outputs=[summary_output, sections_output, raw_output],
            )

        with gr.Tab(translate("config_tab", default_language)):
            config_editor = gr.Code(
                value=_load_config_text(),
                language="json",
                label=translate("config_editor_label", default_language),
                lines=30,
            )
            with gr.Row():
                save_button = gr.Button(translate("config_save", default_language))
                reload_button = gr.Button(translate("config_reload", default_language))
            status_output = gr.Markdown()

            def on_save(config_text: str) -> str:
                return _save_config_from_text(config_text, default_language)

            def on_reload() -> Tuple[str, str]:
                return _load_config_text(), ""

            save_button.click(fn=on_save, inputs=[config_editor], outputs=[status_output])
            reload_button.click(
                fn=on_reload,
                inputs=[],
                outputs=[config_editor, status_output],
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch()
