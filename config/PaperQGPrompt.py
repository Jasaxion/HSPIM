
SECTION_CLASSFICATION_PROMPT = """You are an expert in academic paper analysis. Your task is to classify the sections of a research paper into predefined categories based on their content. The categories are as follows:
1. Abstract
2. Introduction
3. Related Work
4. Approach/Methodology/Model/Method
5. Analysis Theory
6. Experiments
7. Experiment Analysis
8. Discussion/Limitations
9. Conclusion
Please read the provided section text carefully and determine which category it best fits into. If the section does not clearly fit into any of the categories, classify it as 'Other'. Provide **only the category name** as your answer."""

QG_BASE_PROMPT = """You are an experienced academic researcher. Please evaluate the given section of a research paper, focusing on identifying and filtering out any exaggerated claims or one-sided arguments. Specifically, consider the following:
1. The author's claim of novelty does not necessarily indicate true innovation. Extract objective information from the methodology.
2. The author’s claim of significant results may not truly be so. Combine your knowledge and analysis of the experimental data.
3. Provide answers that adhere strictly to the text while incorporating your academic expertise.
Please respond concisely and accurately in scholarly English to the following questions."""

QG_PREDEFINED_QUESTION_COMMON = "Provide arguments for both why this research may lack originality and why it could still be considered novel within its domain."

QG_PREDEFINED_QUESTION_SPECIFIC = {
    "Abstract": "Is there clear evidence to support the contributions as novel?",
    "Introduction": "Is the contribution clearly distinguishable from previous work?",
    "Related Work": "Are there significant differences between this work and others in the field?",
    "Approach/Methodology/Model/Method": "What new methods were used in this research? Do these methods have advantages over existing ones? Please provide supporting cases.",
    "Analysis Theory": "Are the assumptions in the theoretical model different from those used in previous studies?",
    "Experiments": "How well do the results generalize to other settings?",
    "Experiment Analysis": "What new insights are revealed by the analysis of experimental results? Please provide detailed comparisons with past methods using quantitative data.",
    "Discussion/Limitations": "Evaluate whether the limitations significantly hinder the novelty claims.",
    "Conclusion": "What specific innovative findings are emphasized in the research conclusions? Is there clear innovation?"
}

NOVELTY_QA_PROMPT = """
You are an expert academic researcher. Based on the provided section from a research paper, please evaluate it across three dimensions: Novelty, Contribution, and Feasibility.

For each dimension, you must provide:
(i) A Score between 1 and 5 (decimals are allowed).
(ii) A brief Reason justifying your score.
(iii) A Confidence score for your assessment (from 1 to 5), where 5 means you are very confident.

(i) Novelty Score:
5 = Innovative. Groundbreaking work; presents a highly original approach or insight.
4 = Creative. A clever approach or a substantial extension of prior research.
3 = Respectable. A solid, notable extension of existing methods.
2 = Uninspiring. An obvious or very minor improvement on familiar techniques.
1 = Derivative. The core ideas have been done before, and perhaps done better.

(ii) Contribution Score:
5 = Landmark. A major breakthrough that could redefine the field.
4 = Significant. Addresses an important problem and provides impactful results or a valuable new resource.
3 = Useful. A solid, helpful addition that will be used by other researchers.
2 = Incremental. A minor addition with limited potential impact.
1 = Negligible. The work has little to no practical or theoretical impact.

(iii) Feasibility Score:
5 = Highly Feasible. The methods are described clearly, and the work is easily reproducible with standard resources.
4 = Feasible. Can be reproduced with reasonable effort.
3 = Moderately Feasible. Reproduction requires significant effort or specialized, less accessible resources.
2 = Barely Feasible. The methodology is vague, or it relies on proprietary data, making reproduction very difficult.
1 = Infeasible. There is not enough information to reproduce the work.

You MUST provide your response in a single, valid JSON object. Do not include any text outside the JSON structure. Use the following format:
{
    "Novelty": { "Score": float, "Reason": "...", "Confidence": float },
    "Contribution": { "Score": float, "Reason": "...", "Confidence": float },
    "Feasibility": { "Score": float, "Reason": "...", "Confidence": float }
}
"""

SCORING_PROMPT = NOVELTY_QA_PROMPT + "\n(iii) Please output the following format: JSON OUTPUT {Novelty_Score, Reason, Confidence_Score}"

QG_BASIC_PROMPT = {
    "qg_base": QG_BASE_PROMPT,
    "qg_common": QG_PREDEFINED_QUESTION_COMMON,
    "qg_sec": QG_PREDEFINED_QUESTION_SPECIFIC,
    "novelty_qa": NOVELTY_QA_PROMPT
}