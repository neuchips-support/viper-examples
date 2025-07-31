import os
import platform

from transformers import (
    LlamaTokenizer,
    AutoTokenizer,
)

def get_data_path():
    if platform.system() == "Linux":
        return f"/data/"
    elif platform.system() == "Windows":
        if os.path.exists("C:/data/"):
            return f"C:/data/"
        elif os.path.exists("D:/data/"):
            return f"D:/data/"
        else:
            raise FileNotFoundError("Neither C:/data/ nor D:/data/ exists")
    else:
        raise NotImplementedError("Unsupported operating system")

# Define the llm configuration structure
# LLM model
#  -> :path                           # model path
#     :precompiler_cache_path         # model optimize weight path
#     :tokenizer                      # model tokenizer
#     :eos_token                      # model eos token
#     :prompt_template                # model prompt template
#     :preserving_prompt_template     # model context-preserving prompt template
llm_config = {
    "Qwen3-8B": {
        "path": get_data_path() + "/gguf/neuchips/qwen/Qwen3-8B-F16.gguf",
        "precompiler_cache_path": "./data/precompiler_cache/qwen/Qwen3-8B/gguf/",
        "chat_format": "qwen",
        "prompt_template": [
            {"role": "system","content": "[system_setting]",},
            {"role": "user", "content": "[user_prompt_setting]"},
        ],
        "preserving_prompt_template": [
            {"role": "system","content": "[system_setting]",},
            {"role": "user", "content": "[pre_prompt_setting]"},
            {"role": "assistant", "content": "[pre_response_setting]"},
            {"role": "user", "content": "[user_prompt_setting]"},
        ],
    },
    "Qwen2.5-7B": {
        "path": get_data_path() + "/gguf/neuchips/qwen/Qwen2.5-7B-Instruct-F16.gguf",
        "precompiler_cache_path": "./data/precompiler_cache/qwen/Qwen2.5-7B-Instruct/gguf/",
        "chat_format": "qwen",
        "prompt_template": [
            {"role": "system","content": "[system_setting]",},
            {"role": "user", "content": "[user_prompt_setting]"},
        ],
        "preserving_prompt_template": [
            {"role": "system","content": "[system_setting]",},
            {"role": "user", "content": "[pre_prompt_setting]"},
            {"role": "assistant", "content": "[pre_response_setting]"},
            {"role": "user", "content": "[user_prompt_setting]"},
        ],
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "path": get_data_path() + "/gguf/neuchips/DeepSeek-R1-Distill-Llama-8B/DeepSeek-R1-Distill-Llama-8B-F16.gguf",
        "precompiler_cache_path": "./data/precompiler_cache/meta-llama/DeepSeek-R1-Distill-Llama-8B/gguf/",
        "chat_format": "deepseek-r1",
        "prompt_template": [
            {"role": "system","content": "[system_setting]",},
            {"role": "user", "content": "[user_prompt_setting]"},
        ],
        "preserving_prompt_template": [
            {"role": "system","content": "[system_setting]",},
            {"role": "user", "content": "[pre_prompt_setting]"},
            {"role": "assistant", "content": "[pre_response_setting]"},
            {"role": "user", "content": "[user_prompt_setting]"},
        ],
    },
    "Llama-2-7B-Chat": {
        "path": get_data_path() + "/gguf/neuchips/llama2/models/llama-2-7b-chat/ggml-model-f16.gguf",
        "precompiler_cache_path": "./data/precompiler_cache/meta-llama/Llama-2-7b-chat-hf/gguf/",
        "chat_format": "llama-2",
        "prompt_template": [
            {"role": "system","content": "[system_setting]",},
            {"role": "user", "content": "[user_prompt_setting]"},
        ],
        "preserving_prompt_template": [
            {"role": "system","content": "[system_setting]",},
            {"role": "user", "content": "[pre_prompt_setting]"},
            {"role": "assistant", "content": "[pre_response_setting]"},
            {"role": "user", "content": "[user_prompt_setting]"},
        ],
    },
    "Llama-3.1-8B-Instruct": {
        "path": get_data_path() + "/gguf/neuchips/llama3/models/Meta-Llama-3-8B-Instruct/ggml-model-f16.gguf",
        "precompiler_cache_path": "./data/precompiler_cache/meta-llama/Meta-Llama-3.1-8B-Instruct/gguf/",
        "chat_format": "llama-3",
        "prompt_template": [
            {"role": "system","content": "[system_setting]",},
            {"role": "user", "content": "[user_prompt_setting]"},
        ],
        "preserving_prompt_template": [
            {"role": "system","content": "[system_setting]",},
            {"role": "user", "content": "[pre_prompt_setting]"},
            {"role": "assistant", "content": "[pre_response_setting]"},
            {"role": "user", "content": "[user_prompt_setting]"},
        ],
    },
}

# Define the max number of the preserving prompt for next inference
# [TODO] Now only support 0 and 1
max_preserving_prompt_count = 1

# Define the folder to store the rag upload files.
save_path = "./data/uploaded_files/"

# Define the folder to store the rag db data.
rag_db_path = "./data/db/"

# Define the folder to store the rag db bin files.
weight_bin_file_dir = "weight_bin_fils/"

# Define the file name for RAG bin files. (Note: Ensure this file name is synchronized with the Runtime system for consistency.)
rag_db_bin_file_name = "_rag_db.bin"

# Define the logging.
user_activity_logging = False
