"""Simple internationalisation helper for the Gradio interface."""
from __future__ import annotations

from typing import Dict

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "HSPIM - Paper Innovation Analyzer",
        "upload_label": "Upload paper (PDF or MinerU JSON)",
        "enhanced_parsing": "Enable enhanced parsing",
        "language_label": "Interface language",
        "analyze_button": "Analyze",
        "config_tab": "Configuration",
        "analysis_tab": "Analysis",
        "config_editor_label": "Model configuration (editable JSON)",
        "config_save": "Save configuration",
        "config_reload": "Reload",
        "final_score": "Predicted Innovation Score",
        "sections_title": "Section Evaluations",
        "raw_output_title": "Raw Output JSON",
        "error_prefix": "Error",
        "success_config": "Configuration saved successfully.",
        "config_error": "Failed to save configuration: {error}",
        "no_file": "Please upload a paper in PDF or JSON format.",
        "processing": "Processing paper...",
        "mineru_missing": "MinerU API key is required for PDF parsing.",
        "enhanced_enabled": "Enhanced parsing enabled",
        "analysis_summary": "Analysis summary",
        "section_header": "Section",
        "section_question": "Question & Answer",
        "section_scores": "Scores",
        "confidence": "Confidence",
        "reason": "Reason",
        "score": "Score",
        "download_json": "Download JSON"
    },
    "zh": {
        "app_title": "HSPIM - 论文创新性分析",
        "upload_label": "上传论文（PDF 或 MinerU JSON）",
        "enhanced_parsing": "开启增强解析",
        "language_label": "界面语言",
        "analyze_button": "开始分析",
        "config_tab": "配置",
        "analysis_tab": "分析",
        "config_editor_label": "模型配置（可编辑 JSON）",
        "config_save": "保存配置",
        "config_reload": "重新加载",
        "final_score": "预测创新性得分",
        "sections_title": "章节评估",
        "raw_output_title": "原始输出 JSON",
        "error_prefix": "错误",
        "success_config": "配置保存成功。",
        "config_error": "保存配置失败：{error}",
        "no_file": "请上传 PDF 或 JSON 格式的论文。",
        "processing": "论文处理中……",
        "mineru_missing": "解析 PDF 需要提供 MinerU API Key。",
        "enhanced_enabled": "已开启增强解析",
        "analysis_summary": "分析总结",
        "section_header": "章节",
        "section_question": "问答内容",
        "section_scores": "评分",
        "confidence": "置信度",
        "reason": "原因",
        "score": "得分",
        "download_json": "下载 JSON"
    },
}


def translate(key: str, language: str) -> str:
    table = TRANSLATIONS.get(language) or TRANSLATIONS["en"]
    return table.get(key, TRANSLATIONS["en"].get(key, key))
