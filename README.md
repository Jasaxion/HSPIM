# HSPIM
A Hierarchical Framework for Measuring Scientific Paper Innovation via Large Language Models

[![Paper](https://img.shields.io/badge/arXiv-2508.09459-b31b1b.svg)](https://arxiv.org/abs/2504.14620)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()

## Overview

This repository hosts a Gradio powered application that analyses the
innovation of academic papers following the methodology proposed in
[HSPIM](https://arxiv.org/abs/2504.14620). The workflow comprises:

1. **Document Parsing** – MinerU converts PDFs into JSON with optional
   LLM based enhancement for robustness.
2. **Section Understanding** – each section is classified into a
   predefined taxonomy before targeted question answering.
3. **Novelty Scoring** – multi dimensional (Novelty, Contribution,
   Feasibility) evaluation with confidence weighted aggregation.
4. **Presentation** – a bilingual (English/中文) Gradio interface renders
   detailed reasoning together with the final innovation score.

## Getting Started

Install dependencies (a minimal environment is shown below, adjust as
needed for your preferred model providers):

```bash
pip install -r requirements.txt  # create one if necessary
```

Update `config/ModelConfig.py` with your API credentials. This can also
be done inside the application under the **Configuration** tab where the
JSON representation of the config file can be edited and persisted.

Launch the Gradio app:

```bash
python app.py
```

Upload a PDF (MinerU API key required) or a MinerU JSON export, choose
whether to enable enhanced parsing, and trigger the analysis. Results are
displayed section by section with detailed answers and weighted scores
plus the final innovation rating.

## Project Structure

```
config/           Configuration management and prompts
models/           Model abstraction layer (OpenAI/local transformers)
services/         MinerU wrapper, parsing, evaluation pipeline
utils/            Logging helpers
i18n/             Translation tables for the UI
app.py            Gradio application entrypoint
```

## Notes

- MinerU API integration supports polling for remote extraction. When an
  API key is not configured, the application accepts MinerU JSON exports
  directly.
- Enhanced parsing uses an LLM to normalise MinerU output. Disable it to
  minimise token consumption.
- Section evaluation runs concurrently (default 16 workers) to reduce
  latency when analysing long papers.

