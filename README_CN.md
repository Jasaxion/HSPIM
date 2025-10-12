# HSPIM
基于大语言模型的科学论文创新性层次化评估框架  
A Hierarchical Framework for Measuring Scientific Paper Innovation via Large Language Models

[![Paper](https://img.shields.io/badge/arXiv-2508.09459-b31b1b.svg)](https://arxiv.org/abs/2504.14620)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()

## 概述

本项目提供了一个基于 Gradio 的应用，用于按照论文  
[HSPIM](https://arxiv.org/abs/2504.14620) 中提出的方法，对学术论文的创新性进行分析与评估。  
整体工作流程包括：

1. **文档解析（Document Parsing）**：使用 MinerU 将 PDF 文件转换为 JSON，可选择启用基于大语言模型的增强解析以提高鲁棒性。  
2. **章节理解（Section Understanding）**：将论文的各个章节自动分类到预定义的章节类别中，并基于该分类执行针对性问答。  
3. **创新性评分（Novelty Scoring）**：从创新性（Novelty）、贡献度（Contribution）与可行性（Feasibility）三个维度进行多维度评估，并通过置信度加权聚合得到最终创新分数。  

## 使用指南

### 安装依赖

```bash
pip install -r requirements.txt
````

### 配置模型与 API 信息

在 `config/ModelConfig.py` 中填写模型或 API 的相关凭据。
也可以在应用界面的 **Configuration** 标签页中编辑并保存该配置文件的 JSON 内容。

### 启动 Gradio 应用

```bash
python app.py
```

启动后，将打开一个 Web 界面。
上传论文 PDF（需要 MinerU API Key）或 MinerU 导出的 JSON 文件，
可选择是否启用增强解析，然后启动分析流程。
系统将按章节展示模型分析结果、加权得分以及论文的整体创新性评级。

## 说明

* **MinerU 集成**：系统支持 MinerU 的远程提取接口（通过轮询机制）。若未配置 API Key，也可直接上传 MinerU 的 JSON 导出文件。
* **增强解析**：增强模式会调用大语言模型对 MinerU 输出进行结构化标准化处理，以提升解析效果。若需节省 Token，可关闭此功能。
* **并行评估**：章节分析任务默认并行执行（16 个线程），在分析篇幅较长的论文时可显著降低等待时间。

## 后续工作

* 针对不同学科领域的论文，优化 HSPIM 框架下的预定义问答模板（当前版本使用通用模板）。

## 引用
```
@article{tan2025hierarchical,
  title={A Hierarchical Framework for Measuring Scientific Paper Innovation via Large Language Models},
  author={Tan, Hongming and Zhan, Shaoxiong and Jia, Fengwei and Zheng, Hai-Tao and Chan, Wai Kin},
  journal={arXiv preprint arXiv:2504.14620},
  year={2025}
}

