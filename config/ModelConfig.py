from openai import OpenAI

# You need to replace the API_KEY with your own API key and choose the model you want to use
API_KEY = ""

MinerU_API = ""

MODEL_CON = {
    "model_type": "OPENAI",
    "model_name": "gpt-4o-mini"
}

GENERAL_CON = {
    "data_dir": "data",
    "full_paper": False, #Whether process full paper or single section
    "overwrite_existing_files": False,
    "max_workers": 16,
    "retry_limit": 3,
    "review_dir": "data",
    "output_dir": "data"
}

# List of models available
# Todo: 这里应该支持更多模型，例如开源模型，可使用huggingface或者vllm进行加载
MODEL_LIST = {
    "Deepseek": OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com"),
    "OPENAI": OpenAI(api_key=API_KEY)
}

