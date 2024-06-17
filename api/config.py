import os
from pathlib import Path
from typing import Optional, Dict, List, Union

import dotenv
from loguru import logger
from pydantic import BaseModel, Field

from api.common import jsonify, disable_warnings

dotenv.load_dotenv()

disable_warnings(BaseModel)


def get_bool_env(key, default="false"):
    return os.environ.get(key, default).lower() == "true"


def get_env(key, default):
    val = os.environ.get(key, "")
    return val or default


ENGINE = get_env("ENGINE", "default").lower()
TEI_ENDPOINT = get_env("TEI_ENDPOINT", None)
TASKS = get_env("TASKS", "llm").lower().split(",")  # llm, rag

STORAGE_LOCAL_PATH = get_env(
    "STORAGE_LOCAL_PATH",
    os.path.join(Path(__file__).parents[1], "data", "file_storage")
)
os.makedirs(STORAGE_LOCAL_PATH, exist_ok=True)


class BaseSettings(BaseModel):
    """ Settings class. """
    host: Optional[str] = Field(
        default=get_env("HOST", "0.0.0.0"),
        description="Listen address.",
    )
    port: Optional[int] = Field(
        default=int(get_env("PORT", 8000)),
        description="Listen port.",
    )
    api_prefix: Optional[str] = Field(
        default=get_env("API_PREFIX", "/v1"),
        description="API prefix.",
    )
    engine: Optional[str] = Field(
        default=ENGINE,
        description="Choices are ['default', 'vllm'].",
    )
    tasks: Optional[List[str]] = Field(
        default=list(TASKS),
        description="Choices are ['llm', 'rag'].",
    )
    # device related
    device_map: Optional[Union[str, Dict]] = Field(
        default=get_env("DEVICE_MAP", "auto"),
        description="Device map to load the model."
    )
    gpus: Optional[str] = Field(
        default=get_env("GPUS", None),
        description="Specify which gpus to load the model."
    )
    num_gpus: Optional[int] = Field(
        default=int(get_env("NUM_GPUs", 1)),
        ge=0,
        description="How many gpus to load the model."
    )
    activate_inference: Optional[bool] = Field(
        default=get_bool_env("ACTIVATE_INFERENCE", "true"),
        description="Whether to activate inference."
    )
    model_names: Optional[List] = Field(
        default_factory=list,
        description="All available model names"
    )
    # support for api key check
    api_keys: Optional[List[str]] = Field(
        default=get_env("API_KEYS", "").split(",") if get_env("API_KEYS", "") else None,
        description="Support for api key check."
    )


class LLMSettings(BaseModel):
    # model related
    model_name: Optional[str] = Field(
        default=get_env("MODEL_NAME", None),
        description="The name of the model to use for generating completions."
    )
    model_path: Optional[str] = Field(
        default=get_env("MODEL_PATH", None),
        description="The path to the model to use for generating completions."
    )
    dtype: Optional[str] = Field(
        default=get_env("DTYPE", "half"),
        description="Precision dtype."
    )

    # quantize related
    load_in_8bit: Optional[bool] = Field(
        default=get_bool_env("LOAD_IN_8BIT"),
        description="Whether to load the model in 8 bit."
    )
    load_in_4bit: Optional[bool] = Field(
        default=get_bool_env("LOAD_IN_4BIT"),
        description="Whether to load the model in 4 bit."
    )

    # context related
    context_length: Optional[int] = Field(
        default=int(get_env("CONTEXT_LEN", -1)),
        ge=-1,
        description="Context length for generating completions."
    )
    chat_template: Optional[str] = Field(
        default=get_env("PROMPT_NAME", None),
        description="Chat template for generating completions."
    )

    rope_scaling: Optional[str] = Field(
        default=get_env("ROPE_SCALING", None),
        description="RoPE Scaling."
    )
    flash_attn: Optional[bool] = Field(
        default=get_bool_env("FLASH_ATTN", "auto"),
        description="Use flash attention."
    )

    interrupt_requests: Optional[bool] = Field(
        default=get_bool_env("INTERRUPT_REQUESTS", "true"),
        description="Whether to interrupt requests when a new request is received.",
    )


class RAGSettings(BaseModel):
    # embedding related
    embedding_name: Optional[str] = Field(
        default=get_env("EMBEDDING_NAME", None),
        description="The path to the model to use for generating embeddings."
    )
    rerank_name: Optional[str] = Field(
        default=get_env("RERANK_NAME", None),
        description="The path to the model to use for reranking."
    )
    embedding_size: Optional[int] = Field(
        default=int(get_env("EMBEDDING_SIZE", -1)),
        description="The embedding size to use for generating embeddings."
    )
    embedding_device: Optional[str] = Field(
        default=get_env("EMBEDDING_DEVICE", "cuda:0"),
        description="Device to load the model."
    )
    rerank_device: Optional[str] = Field(
        default=get_env("RERANK_DEVICE", "cuda:0"),
        description="Device to load the model."
    )


class VLLMSetting(BaseModel):
    trust_remote_code: Optional[bool] = Field(
        default=get_bool_env("TRUST_REMOTE_CODE"),
        description="Whether to use remote code."
    )
    tokenize_mode: Optional[str] = Field(
        default=get_env("TOKENIZE_MODE", "auto"),
        description="Tokenize mode for vllm server."
    )
    tensor_parallel_size: Optional[int] = Field(
        default=int(get_env("TENSOR_PARALLEL_SIZE", 1)),
        ge=1,
        description="Tensor parallel size for vllm server."
    )
    gpu_memory_utilization: Optional[float] = Field(
        default=float(get_env("GPU_MEMORY_UTILIZATION", 0.9)),
        description="GPU memory utilization for vllm server."
    )
    max_num_batched_tokens: Optional[int] = Field(
        default=int(get_env("MAX_NUM_BATCHED_TOKENS", -1)),
        ge=-1,
        description="Max num batched tokens for vllm server."
    )
    max_num_seqs: Optional[int] = Field(
        default=int(get_env("MAX_NUM_SEQS", 256)),
        ge=1,
        description="Max num seqs for vllm server."
    )
    quantization_method: Optional[str] = Field(
        default=get_env("QUANTIZATION_METHOD", None),
        description="Quantization method for vllm server."
    )
    enforce_eager: Optional[bool] = Field(
        default=get_bool_env("ENFORCE_EAGER"),
        description="Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid for maximal performance and flexibility."
    )
    max_seq_len_to_capture: Optional[int] = Field(
        default=int(get_env("MAX_SEQ_LEN_TO_CAPTURE", 8192)),
        description="Maximum context length covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode."
    )
    max_loras: Optional[int] = Field(
        default=int(get_env("MAX_LORAS", 1)),
        description="Max number of LoRAs in a single batch."
    )
    max_lora_rank: Optional[int] = Field(
        default=int(get_env("MAX_LORA_RANK", 32)),
        description="Max LoRA rank."
    )
    lora_extra_vocab_size: Optional[int] = Field(
        default=int(get_env("LORA_EXTRA_VOCAB_SIZE", 256)),
        description="Maximum size of extra vocabulary that can be present in a LoRA adapter added to the base model vocabulary."
    )
    lora_dtype: Optional[str] = Field(
        default=get_env("LORA_DTYPE", "auto"),
        description="Data type for LoRA. If auto, will default to base model dtype."
    )
    max_cpu_loras: Optional[int] = Field(
        default=int(get_env("MAX_CPU_LORAS", -1)),
        ge=-1,
    )
    lora_modules: Optional[str] = Field(
        default=get_env("LORA_MODULES", ""),
    )
    disable_custom_all_reduce: Optional[bool] = Field(
        default=get_bool_env("DISABLE_CUSTOM_ALL_REDUCE"),
    )
    vllm_disable_log_stats: Optional[bool] = Field(
        default=get_bool_env("VLLM_DISABLE_LOG_STATS", "true"),
    )
    distributed_executor_backend: Optional[str] = Field(
        default=get_env("DISTRIBUTED_EXECUTOR_BACKEND", None),
    )


TEXT_SPLITTER_CONFIG = {
    "ChineseRecursiveTextSplitter": {
        "source": "huggingface",   # 选择tiktoken则使用openai的方法
        "tokenizer_name_or_path": get_env("EMBEDDING_NAME", ""),
    },
    "SpacyTextSplitter": {
        "source": "huggingface",
        "tokenizer_name_or_path": "gpt2",
    },
    "RecursiveCharacterTextSplitter": {
        "source": "tiktoken",
        "tokenizer_name_or_path": "cl100k_base",
    },
    "MarkdownHeaderTextSplitter": {
        "headers_to_split_on":
            [
                ("#", "head1"),
                ("##", "head2"),
                ("###", "head3"),
                ("####", "head4"),
            ]
    },
}


PARENT_CLASSES = [BaseSettings]

if "llm" in TASKS:
    if ENGINE == "default":
        PARENT_CLASSES.append(LLMSettings)
    elif ENGINE == "vllm":
        PARENT_CLASSES.extend([LLMSettings, VLLMSetting])

if "rag" in TASKS:
    PARENT_CLASSES.append(RAGSettings)


class Settings(*PARENT_CLASSES):
    ...


SETTINGS = Settings()
for name in ["model_name", "embedding_name", "rerank_name"]:
    if getattr(SETTINGS, name, None):
        SETTINGS.model_names.append(getattr(SETTINGS, name).split("/")[-1])
logger.debug(f"SETTINGS: {jsonify(SETTINGS, indent=4)}")


if SETTINGS.gpus:
    if len(SETTINGS.gpus.split(",")) < SETTINGS.num_gpus:
        raise ValueError(
            f"Larger --num_gpus ({SETTINGS.num_gpus}) than --gpus {SETTINGS.gpus}!"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = SETTINGS.gpus
