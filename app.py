"""Gradio application for the HSPIM paper innovation analysis."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr

from config.config_manager import ConfigError, load_config, save_config
from i18n import translate
from services.pipeline import PipelineError, analyse_paper
from services.result_formatter import build_section_html, build_summary_markdown

LANGUAGE_CODES = ["en", "zh"]
LANGUAGE_LABELS = {
    "en": {"en": "English", "zh": "英语"},
    "zh": {"en": "Chinese", "zh": "中文"},
}
THEME_CHOICES = ["soft", "default", "glass", "monochrome"]
APP_STYLE = """
<style>
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.75rem;
}
.app-title h1, .app-title h2, .app-title h3 {
    margin: 0;
}
.app-title {
    flex: 1;
    font-size: 1.9rem !important;
    font-weight: 600 !important;
}
.language-dropdown {
    max-width: 220px;
}
.section-results {
    display: grid;
    gap: 1.1rem;
}
.section-item {
    background: var(--block-background-fill);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    box-shadow: var(--shadow-drop-lg);
    border: 1px solid color-mix(in srgb, var(--color-accent) 20%, transparent);
}
.section-item h3 {
    margin-bottom: 0.5rem;
}
.section-answer {
    line-height: 1.5;
}
.section-scores-title {
    margin-top: 1rem;
}
.section-item ul {
    list-style: none;
    padding-left: 0;
    margin: 0;
}
.section-item li {
    margin-bottom: 0.5rem;
    padding: 0.45rem 0.65rem;
    border-radius: 8px;
    background: color-mix(in srgb, var(--color-primary) 10%, transparent);
}
.results-column {
    gap: 0.8rem;
}
.config-help {
    margin-bottom: 0.5rem;
}
.config-status {
    min-height: 1.5rem;
}
</style>
"""

def _language_label(code: str, display_language: str) -> str:
    labels = LANGUAGE_LABELS.get(code, {})
    return labels.get(display_language, code)

def _language_from_label(label: str) -> str:
    for code, labels in LANGUAGE_LABELS.items():
        if label in labels.values():
            return code
    if label in LANGUAGE_CODES:
        return label
    return LANGUAGE_CODES[0]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)

def _format_extra_field(extra: Any) -> str:
    if isinstance(extra, dict):
        return json.dumps(extra, indent=2, ensure_ascii=False)
    if isinstance(extra, str) and extra.strip():
        try:
            parsed = json.loads(extra)
        except json.JSONDecodeError:
            return extra
        if isinstance(parsed, dict):
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        return extra
    return "{}"

def _parse_extra_field(extra_text: str) -> Dict[str, Any]:
    if not extra_text.strip():
        return {}
    try:
        parsed = json.loads(extra_text)
    except json.JSONDecodeError as exc:
        raise ConfigError("Model extra parameters must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ConfigError("Model extra parameters must be a JSON object.")
    return parsed

def _config_form_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = config.get("model", {})
    general_cfg = config.get("general", {})
    mineru_cfg = config.get("mineru", {})
    ui_cfg = config.get("ui", {})

    return {
        "model_provider": model_cfg.get("provider", "OPENAI"),
        "model_name": model_cfg.get("name", ""),
        "model_api_key": model_cfg.get("api_key", ""),
        "model_base_url": model_cfg.get("base_url", ""),
        "model_local_path": model_cfg.get("local_path", ""),
        "model_engine": (model_cfg.get("engine") or "transformers").lower(),
        "model_tensor_parallel": _as_int(model_cfg.get("tensor_parallel_size", 1)),
        "model_dtype": model_cfg.get("dtype", "auto"),
        "model_trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "model_gpu_utilization": _as_float(model_cfg.get("gpu_memory_utilization", 0.9)),
        "model_chat_template": model_cfg.get("chat_template", ""),
        "model_temperature": _as_float(model_cfg.get("temperature", 0.0)),
        "model_max_tokens": _as_int(model_cfg.get("max_tokens", 0)),
        "model_request_timeout": _as_int(model_cfg.get("request_timeout", 60)),
        "model_extra": _format_extra_field(model_cfg.get("extra", {})),
        "general_data_dir": general_cfg.get("data_dir", ""),
        "general_review_dir": general_cfg.get("review_dir", ""),
        "general_output_dir": general_cfg.get("output_dir", ""),
        "general_overwrite": bool(general_cfg.get("overwrite_existing_files", False)),
        "general_max_workers": _as_int(general_cfg.get("max_workers", 1)),
        "general_retry_limit": _as_int(general_cfg.get("retry_limit", 0)),
        "general_enable_logging": bool(general_cfg.get("enable_logging", False)),
        "mineru_api_key": mineru_cfg.get("api_key", ""),
        "mineru_base_url": mineru_cfg.get("base_url", ""),
        "mineru_use_ocr": bool(mineru_cfg.get("use_ocr", False)),
        "mineru_enable_formula": bool(mineru_cfg.get("enable_formula", False)),
        "mineru_enable_table": bool(mineru_cfg.get("enable_table", False)),
        "mineru_language": mineru_cfg.get("language", ""),
        "ui_default_language": ui_cfg.get("default_language", LANGUAGE_CODES[0]),
        "ui_theme": ui_cfg.get("theme", THEME_CHOICES[0]),
        "ui_enhanced_parsing_default": bool(ui_cfg.get("enhanced_parsing_default", False)),
    }

def _build_config_payload(base_config: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.loads(json.dumps(base_config))
    base_model = base_config.get("model", {})
    base_general = base_config.get("general", {})
    model_extra = _parse_extra_field(values["model_extra"])
    payload["model"] = {
        "provider": values["model_provider"],
        "name": values["model_name"],
        "api_key": values["model_api_key"],
        "base_url": values["model_base_url"],
        "local_path": values["model_local_path"],
        "engine": (values["model_engine"] or "transformers").lower(),
        "tensor_parallel_size": _as_int(
            values["model_tensor_parallel"], base_model.get("tensor_parallel_size", 1)
        ),
        "dtype": values["model_dtype"],
        "trust_remote_code": bool(values["model_trust_remote_code"]),
        "gpu_memory_utilization": _as_float(
            values["model_gpu_utilization"], base_model.get("gpu_memory_utilization", 0.9)
        ),
        "chat_template": values["model_chat_template"],
        "temperature": _as_float(values["model_temperature"], base_model.get("temperature", 0.0)),
        "max_tokens": _as_int(values["model_max_tokens"], base_model.get("max_tokens", 0)),
        "request_timeout": _as_int(values["model_request_timeout"], base_model.get("request_timeout", 60)),
        "extra": model_extra,
    }
    payload["general"] = {
        "data_dir": values["general_data_dir"],
        "review_dir": values["general_review_dir"],
        "output_dir": values["general_output_dir"],
        "overwrite_existing_files": bool(values["general_overwrite"]),
        "max_workers": _as_int(values["general_max_workers"], base_general.get("max_workers", 1)),
        "retry_limit": _as_int(values["general_retry_limit"], base_general.get("retry_limit", 0)),
        "enable_logging": bool(values["general_enable_logging"]),
    }
    payload["mineru"] = {
        "api_key": values["mineru_api_key"],
        "base_url": values["mineru_base_url"],
        "use_ocr": bool(values["mineru_use_ocr"]),
        "enable_formula": bool(values["mineru_enable_formula"]),
        "enable_table": bool(values["mineru_enable_table"]),
        "language": values["mineru_language"],
    }
    chosen_language = values["ui_default_language"]
    if chosen_language not in LANGUAGE_CODES:
        chosen_language = base_config.get("ui", {}).get("default_language", LANGUAGE_CODES[0])
        if chosen_language not in LANGUAGE_CODES:
            chosen_language = LANGUAGE_CODES[0]
    chosen_theme = values["ui_theme"] or base_config.get("ui", {}).get("theme", THEME_CHOICES[0])
    payload["ui"] = {
        "default_language": chosen_language,
        "theme": chosen_theme,
        "enhanced_parsing_default": bool(values["ui_enhanced_parsing_default"]),
    }
    return payload

def _language_choices(current_language: str) -> list[str]:
    return [_language_label(code, current_language) for code in LANGUAGE_CODES]

async def _run_analysis(file_obj, use_enhanced: bool, language: str) -> Tuple[str, str, Dict[str, object]]:
    language = language if language in LANGUAGE_CODES else LANGUAGE_CODES[0]
    if file_obj is None:
        return translate("no_file", language), "", {}
    temp_dir = Path(tempfile.mkdtemp())
    uploaded_path = temp_dir / Path(file_obj.name).name
    shutil.copy(file_obj.name, uploaded_path)
    try:
        result = await analyse_paper(uploaded_path, use_enhanced=use_enhanced)
    except PipelineError as exc:
        return f"**{translate('error_prefix', language)}:** {exc}", "", {}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    summary = build_summary_markdown(result, language)
    sections_html = build_section_html(result.get("evaluations", []), language)
    return summary, sections_html, result

def build_app() -> gr.Blocks:
    config = load_config()
    ui_config = config.get("ui", {})
    default_language = ui_config.get("default_language", LANGUAGE_CODES[0])
    if default_language not in LANGUAGE_CODES:
        default_language = LANGUAGE_CODES[0]
    theme = ui_config.get("theme", THEME_CHOICES[0])
    if theme not in THEME_CHOICES:
        theme = THEME_CHOICES[0]
    enhanced_default = ui_config.get("enhanced_parsing_default", False)
    form_defaults = _config_form_defaults(config)
    provider_choices = list(config.get("available_models", {}).keys()) or ["OPENAI", "TRANSFORMERS"]
    theme_options = list(dict.fromkeys([*THEME_CHOICES, form_defaults["ui_theme"], theme]))

    with gr.Blocks(theme=theme, title=translate("app_title", default_language)) as demo:
        gr.HTML(APP_STYLE)
        language_state = gr.State(default_language)

        with gr.Row(elem_classes=["app-header"]):
            title_md = gr.Markdown(
                f"# {translate('app_title', default_language)}",
                elem_classes=["app-title"],
            )
            language_dropdown = gr.Dropdown(
                choices=_language_choices(default_language),
                value=_language_label(default_language, default_language),
                label=translate("language_label", default_language),
                interactive=True,
                elem_classes=["language-dropdown"],
            )

        with gr.Tabs() as tabs:
            with gr.Tab(translate("analysis_tab", default_language)) as analysis_tab:
                with gr.Row():
                    file_input = gr.File(label=translate("upload_label", default_language))
                    enhanced_toggle = gr.Checkbox(
                        value=enhanced_default,
                        label=translate("enhanced_parsing", default_language),
                    )
                analyze_button = gr.Button(
                    value=translate("analyze_button", default_language),
                    variant="primary",
                )
                summary_output = gr.Markdown(
                    label=translate("analysis_summary", default_language),
                    elem_classes=["results-column"],
                )
                sections_output = gr.HTML(label=translate("sections_title", default_language))
                raw_output = gr.JSON(label=translate("raw_output_title", default_language))

                async def on_analyze(file_obj, enhanced, language_value):
                    lang_code = _language_from_label(language_value) if language_value else default_language
                    return await _run_analysis(file_obj, enhanced, lang_code)

                analyze_button.click(
                    fn=on_analyze,
                    inputs=[file_input, enhanced_toggle, language_dropdown],
                    outputs=[summary_output, sections_output, raw_output],
                )

            with gr.Tab(translate("config_tab", default_language)) as config_tab:
                config_intro = gr.Markdown(
                    translate("config_description", default_language),
                    elem_classes=["config-help"],
                )
                with gr.Accordion(translate("model_section_title", default_language), open=True) as model_section:
                    model_provider = gr.Dropdown(
                        choices=provider_choices,
                        value=form_defaults["model_provider"],
                        label=translate("model_provider", default_language),
                    )
                    model_name = gr.Textbox(
                        value=form_defaults["model_name"],
                        label=translate("model_name", default_language),
                    )

                    with gr.Group(visible=form_defaults["model_provider"].upper() == "OPENAI") as openai_group:
                        model_api_key = gr.Textbox(
                            value=form_defaults["model_api_key"],
                            label=translate("model_api_key", default_language),
                            type="password",
                        )
                        model_base_url = gr.Textbox(
                            value=form_defaults["model_base_url"],
                            label=translate("model_base_url", default_language),
                        )
                        model_request_timeout = gr.Number(
                            value=form_defaults["model_request_timeout"],
                            label=translate("model_request_timeout", default_language),
                            precision=0,
                        )

                    with gr.Group(visible=form_defaults["model_provider"].upper() == "TRANSFORMERS") as local_group:
                        model_local_path = gr.Textbox(
                            value=form_defaults["model_local_path"],
                            label=translate("model_local_path", default_language),
                            placeholder="/path/to/model-or-hub-id",
                        )
                        model_engine = gr.Dropdown(
                            choices=["transformers", "vllm"],
                            value=(form_defaults["model_engine"] or "transformers").lower(),
                            label=translate("model_engine", default_language),
                        )
                        model_tensor_parallel = gr.Number(
                            value=form_defaults["model_tensor_parallel"],
                            label=translate("model_tensor_parallel", default_language),
                            precision=0,
                        )
                        model_dtype = gr.Textbox(
                            value=form_defaults["model_dtype"],
                            label=translate("model_dtype", default_language),
                        )
                        model_trust_remote_code = gr.Checkbox(
                            value=form_defaults["model_trust_remote_code"],
                            label=translate("model_trust_remote_code", default_language),
                        )
                        model_gpu_utilization = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            step=0.05,
                            value=form_defaults["model_gpu_utilization"],
                            label=translate("model_gpu_utilization", default_language),
                        )
                        model_chat_template = gr.Textbox(
                            value=form_defaults["model_chat_template"],
                            label=translate("model_chat_template", default_language),
                            lines=4,
                        )

                    model_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        value=form_defaults["model_temperature"],
                        label=translate("model_temperature", default_language),
                    )
                    model_max_tokens = gr.Number(
                        value=form_defaults["model_max_tokens"],
                        label=translate("model_max_tokens", default_language),
                        precision=0,
                    )
                    model_extra = gr.Textbox(
                        value=form_defaults["model_extra"],
                        label=translate("model_extra", default_language),
                        lines=4,
                    )

                with gr.Accordion(translate("general_section_title", default_language), open=False) as general_section:
                    general_data_dir = gr.Textbox(
                        value=form_defaults["general_data_dir"],
                        label=translate("general_data_dir", default_language),
                    )
                    general_review_dir = gr.Textbox(
                        value=form_defaults["general_review_dir"],
                        label=translate("general_review_dir", default_language),
                    )
                    general_output_dir = gr.Textbox(
                        value=form_defaults["general_output_dir"],
                        label=translate("general_output_dir", default_language),
                    )
                    general_overwrite = gr.Checkbox(
                        value=form_defaults["general_overwrite"],
                        label=translate("general_overwrite", default_language),
                    )
                    general_max_workers = gr.Number(
                        value=form_defaults["general_max_workers"],
                        label=translate("general_max_workers", default_language),
                        precision=0,
                    )
                    general_retry_limit = gr.Number(
                        value=form_defaults["general_retry_limit"],
                        label=translate("general_retry_limit", default_language),
                        precision=0,
                    )
                    general_enable_logging = gr.Checkbox(
                        value=form_defaults["general_enable_logging"],
                        label=translate("general_enable_logging", default_language),
                    )

                with gr.Accordion(translate("mineru_section_title", default_language), open=False) as mineru_section:
                    mineru_api_key = gr.Textbox(
                        value=form_defaults["mineru_api_key"],
                        label=translate("mineru_api_key", default_language),
                        type="password",
                    )
                    mineru_base_url = gr.Textbox(
                        value=form_defaults["mineru_base_url"],
                        label=translate("mineru_base_url", default_language),
                    )
                    mineru_use_ocr = gr.Checkbox(
                        value=form_defaults["mineru_use_ocr"],
                        label=translate("mineru_use_ocr", default_language),
                    )
                    mineru_enable_formula = gr.Checkbox(
                        value=form_defaults["mineru_enable_formula"],
                        label=translate("mineru_enable_formula", default_language),
                    )
                    mineru_enable_table = gr.Checkbox(
                        value=form_defaults["mineru_enable_table"],
                        label=translate("mineru_enable_table", default_language),
                    )
                    mineru_language = gr.Textbox(
                        value=form_defaults["mineru_language"],
                        label=translate("mineru_language", default_language),
                    )

                with gr.Accordion(translate("ui_section_title", default_language), open=False) as ui_section:
                    ui_default_language = gr.Dropdown(
                        choices=LANGUAGE_CODES,
                        value=form_defaults["ui_default_language"],
                        label=translate("ui_default_language", default_language),
                    )
                    ui_theme = gr.Dropdown(
                        choices=theme_options,
                        value=form_defaults["ui_theme"],
                        label=translate("ui_theme", default_language),
                    )
                    ui_enhanced_parsing_default = gr.Checkbox(
                        value=form_defaults["ui_enhanced_parsing_default"],
                        label=translate("ui_enhanced_parsing_default", default_language),
                    )

                with gr.Row():
                    save_button = gr.Button(
                        value=translate("config_save", default_language),
                        variant="primary",
                    )
                    reload_button = gr.Button(
                        value=translate("config_reload", default_language)
                    )
                status_output = gr.Markdown(elem_classes=["config-status"])

                def on_save(
                    language_value,
                    model_provider_value,
                    model_name_value,
                    model_api_key_value,
                    model_base_url_value,
                    model_local_path_value,
                    model_engine_value,
                    model_tensor_parallel_value,
                    model_dtype_value,
                    model_trust_remote_code_value,
                    model_gpu_utilization_value,
                    model_chat_template_value,
                    model_temperature_value,
                    model_max_tokens_value,
                    model_request_timeout_value,
                    model_extra_value,
                    general_data_dir_value,
                    general_review_dir_value,
                    general_output_dir_value,
                    general_overwrite_value,
                    general_max_workers_value,
                    general_retry_limit_value,
                    general_enable_logging_value,
                    mineru_api_key_value,
                    mineru_base_url_value,
                    mineru_use_ocr_value,
                    mineru_enable_formula_value,
                    mineru_enable_table_value,
                    mineru_language_value,
                    ui_default_language_value,
                    ui_theme_value,
                    ui_enhanced_default_value,
                ) -> str:
                    lang_code = language_value if language_value in LANGUAGE_CODES else default_language
                    try:
                        payload = _build_config_payload(
                            load_config(),
                            {
                                "model_provider": model_provider_value,
                                "model_name": model_name_value,
                                "model_api_key": model_api_key_value,
                                "model_base_url": model_base_url_value,
                                "model_local_path": model_local_path_value,
                                "model_engine": model_engine_value,
                                "model_tensor_parallel": model_tensor_parallel_value,
                                "model_dtype": model_dtype_value,
                                "model_trust_remote_code": model_trust_remote_code_value,
                                "model_gpu_utilization": model_gpu_utilization_value,
                                "model_chat_template": model_chat_template_value,
                                "model_temperature": model_temperature_value,
                                "model_max_tokens": model_max_tokens_value,
                                "model_request_timeout": model_request_timeout_value,
                                "model_extra": model_extra_value,
                                "general_data_dir": general_data_dir_value,
                                "general_review_dir": general_review_dir_value,
                                "general_output_dir": general_output_dir_value,
                                "general_overwrite": general_overwrite_value,
                                "general_max_workers": general_max_workers_value,
                                "general_retry_limit": general_retry_limit_value,
                                "general_enable_logging": general_enable_logging_value,
                                "mineru_api_key": mineru_api_key_value,
                                "mineru_base_url": mineru_base_url_value,
                                "mineru_use_ocr": mineru_use_ocr_value,
                                "mineru_enable_formula": mineru_enable_formula_value,
                                "mineru_enable_table": mineru_enable_table_value,
                                "mineru_language": mineru_language_value,
                                "ui_default_language": ui_default_language_value,
                                "ui_theme": ui_theme_value,
                                "ui_enhanced_parsing_default": ui_enhanced_default_value,
                            },
                        )
                        save_config(payload)
                    except ConfigError as exc:
                        return translate("config_error", lang_code).format(error=exc)
                    return translate("success_config", lang_code)

                def on_reload(language_value):
                    lang_code = language_value if language_value in LANGUAGE_CODES else default_language
                    latest = _config_form_defaults(load_config())
                    provider = (latest["model_provider"] or "").upper()
                    return (
                        gr.update(value=latest["model_provider"]),
                        gr.update(value=latest["model_name"]),
                        gr.update(value=latest["model_api_key"]),
                        gr.update(value=latest["model_base_url"]),
                        gr.update(value=latest["model_local_path"]),
                        gr.update(value=(latest["model_engine"] or "transformers").lower()),
                        gr.update(value=latest["model_tensor_parallel"]),
                        gr.update(value=latest["model_dtype"]),
                        gr.update(value=latest["model_trust_remote_code"]),
                        gr.update(value=latest["model_gpu_utilization"]),
                        gr.update(value=latest["model_chat_template"]),
                        gr.update(value=latest["model_temperature"]),
                        gr.update(value=latest["model_max_tokens"]),
                        gr.update(value=latest["model_request_timeout"]),
                        gr.update(value=latest["model_extra"]),
                        gr.update(value=latest["general_data_dir"]),
                        gr.update(value=latest["general_review_dir"]),
                        gr.update(value=latest["general_output_dir"]),
                        gr.update(value=latest["general_overwrite"]),
                        gr.update(value=latest["general_max_workers"]),
                        gr.update(value=latest["general_retry_limit"]),
                        gr.update(value=latest["general_enable_logging"]),
                        gr.update(value=latest["mineru_api_key"]),
                        gr.update(value=latest["mineru_base_url"]),
                        gr.update(value=latest["mineru_use_ocr"]),
                        gr.update(value=latest["mineru_enable_formula"]),
                        gr.update(value=latest["mineru_enable_table"]),
                        gr.update(value=latest["mineru_language"]),
                        gr.update(value=latest["ui_default_language"]),
                        gr.update(value=latest["ui_theme"]),
                        gr.update(value=latest["ui_enhanced_parsing_default"]),
                        gr.update(visible=provider == "OPENAI"),
                        gr.update(visible=provider == "TRANSFORMERS"),
                        gr.update(value=translate("config_reload_success", lang_code)),
                    )

                def on_provider_change(provider_value):
                    provider = (provider_value or "").upper()
                    return (
                        gr.update(visible=provider == "OPENAI"),
                        gr.update(visible=provider == "TRANSFORMERS"),
                    )

                save_button.click(
                    fn=on_save,
                    inputs=[
                        language_state,
                        model_provider,
                        model_name,
                        model_api_key,
                        model_base_url,
                        model_local_path,
                        model_engine,
                        model_tensor_parallel,
                        model_dtype,
                        model_trust_remote_code,
                        model_gpu_utilization,
                        model_chat_template,
                        model_temperature,
                        model_max_tokens,
                        model_request_timeout,
                        model_extra,
                        general_data_dir,
                        general_review_dir,
                        general_output_dir,
                        general_overwrite,
                        general_max_workers,
                        general_retry_limit,
                        general_enable_logging,
                        mineru_api_key,
                        mineru_base_url,
                        mineru_use_ocr,
                        mineru_enable_formula,
                        mineru_enable_table,
                        mineru_language,
                        ui_default_language,
                        ui_theme,
                        ui_enhanced_parsing_default,
                    ],
                    outputs=[status_output],
                )

                reload_button.click(
                    fn=on_reload,
                    inputs=[language_state],
                    outputs=[
                        model_provider,
                        model_name,
                        model_api_key,
                        model_base_url,
                        model_local_path,
                        model_engine,
                        model_tensor_parallel,
                        model_dtype,
                        model_trust_remote_code,
                        model_gpu_utilization,
                        model_chat_template,
                        model_temperature,
                        model_max_tokens,
                        model_request_timeout,
                        model_extra,
                        general_data_dir,
                        general_review_dir,
                        general_output_dir,
                        general_overwrite,
                        general_max_workers,
                        general_retry_limit,
                        general_enable_logging,
                        mineru_api_key,
                        mineru_base_url,
                        mineru_use_ocr,
                        mineru_enable_formula,
                        mineru_enable_table,
                        mineru_language,
                        ui_default_language,
                        ui_theme,
                        ui_enhanced_parsing_default,
                        openai_group,
                        local_group,
                        status_output,
                    ],
                )

                model_provider.change(
                    fn=on_provider_change,
                    inputs=[model_provider],
                    outputs=[openai_group, local_group],
                )

                def on_language_change(selection: str):
                    lang_code = _language_from_label(selection)
                    choices = _language_choices(lang_code)
                    return (
                        lang_code,
                        gr.update(value=f"# {translate('app_title', lang_code)}"),
                        gr.update(
                            choices=choices,
                            value=_language_label(lang_code, lang_code),
                            label=translate("language_label", lang_code),
                        ),
                        gr.update(label=translate("analysis_tab", lang_code)),
                        gr.update(label=translate("config_tab", lang_code)),
                        gr.update(label=translate("upload_label", lang_code)),
                        gr.update(label=translate("enhanced_parsing", lang_code)),
                        gr.update(value=translate("analyze_button", lang_code)),
                        gr.update(label=translate("analysis_summary", lang_code)),
                        gr.update(label=translate("sections_title", lang_code)),
                        gr.update(label=translate("raw_output_title", lang_code)),
                        gr.update(value=translate("config_description", lang_code)),
                        gr.update(label=translate("model_section_title", lang_code)),
                        gr.update(label=translate("model_provider", lang_code)),
                        gr.update(label=translate("model_name", lang_code)),
                        gr.update(label=translate("model_api_key", lang_code)),
                        gr.update(label=translate("model_base_url", lang_code)),
                        gr.update(label=translate("model_local_path", lang_code)),
                        gr.update(label=translate("model_engine", lang_code)),
                        gr.update(label=translate("model_tensor_parallel", lang_code)),
                        gr.update(label=translate("model_dtype", lang_code)),
                        gr.update(label=translate("model_trust_remote_code", lang_code)),
                        gr.update(label=translate("model_gpu_utilization", lang_code)),
                        gr.update(label=translate("model_chat_template", lang_code)),
                        gr.update(label=translate("model_temperature", lang_code)),
                        gr.update(label=translate("model_max_tokens", lang_code)),
                        gr.update(label=translate("model_request_timeout", lang_code)),
                        gr.update(label=translate("model_extra", lang_code)),
                        gr.update(label=translate("general_section_title", lang_code)),
                        gr.update(label=translate("general_data_dir", lang_code)),
                        gr.update(label=translate("general_review_dir", lang_code)),
                        gr.update(label=translate("general_output_dir", lang_code)),
                        gr.update(label=translate("general_overwrite", lang_code)),
                        gr.update(label=translate("general_max_workers", lang_code)),
                        gr.update(label=translate("general_retry_limit", lang_code)),
                        gr.update(label=translate("general_enable_logging", lang_code)),
                        gr.update(label=translate("mineru_section_title", lang_code)),
                        gr.update(label=translate("mineru_api_key", lang_code)),
                        gr.update(label=translate("mineru_base_url", lang_code)),
                        gr.update(label=translate("mineru_use_ocr", lang_code)),
                        gr.update(label=translate("mineru_enable_formula", lang_code)),
                        gr.update(label=translate("mineru_enable_table", lang_code)),
                        gr.update(label=translate("mineru_language", lang_code)),
                        gr.update(label=translate("ui_section_title", lang_code)),
                        gr.update(label=translate("ui_default_language", lang_code)),
                        gr.update(label=translate("ui_theme", lang_code)),
                        gr.update(label=translate("ui_enhanced_parsing_default", lang_code)),
                        gr.update(value=translate("config_save", lang_code)),
                        gr.update(value=translate("config_reload", lang_code)),
                        gr.update(value=""),
                    )

            language_dropdown.change(
                fn=on_language_change,
                inputs=[language_dropdown],
                outputs=[
                    language_state,
                    title_md,
                    language_dropdown,
                    analysis_tab,
                    config_tab,
                    file_input,
                    enhanced_toggle,
                    analyze_button,
                    summary_output,
                    sections_output,
                    raw_output,
                    config_intro,
                    model_section,
                    model_provider,
                    model_name,
                    model_api_key,
                    model_base_url,
                    model_local_path,
                    model_engine,
                    model_tensor_parallel,
                    model_dtype,
                    model_trust_remote_code,
                    model_gpu_utilization,
                    model_chat_template,
                    model_temperature,
                    model_max_tokens,
                    model_request_timeout,
                    model_extra,
                    general_section,
                    general_data_dir,
                    general_review_dir,
                    general_output_dir,
                    general_overwrite,
                    general_max_workers,
                    general_retry_limit,
                    general_enable_logging,
                    mineru_section,
                    mineru_api_key,
                    mineru_base_url,
                    mineru_use_ocr,
                    mineru_enable_formula,
                    mineru_enable_table,
                    mineru_language,
                    ui_section,
                    ui_default_language,
                    ui_theme,
                    ui_enhanced_parsing_default,
                    save_button,
                    reload_button,
                    status_output,
            ],
        )

    return demo

if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch()
