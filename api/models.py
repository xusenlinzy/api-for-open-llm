import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.apapter import get_prompt_adapter
from api.config import SETTINGS


def create_app():
    """ create fastapi app server """
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def create_embedding_model():
    """ get embedding model from sentence-transformers. """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SETTINGS.embedding_name, device=SETTINGS.embedding_device)


def create_generate_model():
    """ get generate model for chat or completion. """
    from api.generation import ModelServer
    from api.apapter.model import load_model

    if SETTINGS.patch_type == "attention":
        from api.utils.patches import apply_attention_patch

        apply_attention_patch(use_memory_efficient_attention=True)
    if SETTINGS.patch_type == "ntk":
        from api.utils.patches import apply_ntk_scaling_patch

        apply_ntk_scaling_patch(SETTINGS.alpha)

    include = {
        "model_name", "quantize", "device", "device_map", "num_gpus",
        "load_in_8bit", "load_in_4bit", "using_ptuning_v2", "dtype", "resize_embeddings"
    }
    kwargs = SETTINGS.dict(include=include)

    model, tokenizer = load_model(
        model_name_or_path=SETTINGS.model_path,
        adapter_model=SETTINGS.adapter_model_path,
        **kwargs,
    )

    logger.info("Using default engine")

    return ModelServer(
        model,
        tokenizer,
        SETTINGS.device,
        model_name=SETTINGS.model_name,
        context_len=SETTINGS.context_length if SETTINGS.context_length > 0 else None,
        stream_interval=SETTINGS.stream_interverl,
        prompt_name=SETTINGS.chat_template,
        use_streamer_v2=SETTINGS.use_streamer_v2,
    )


def get_context_len(model_config) -> int:
    """ fix for model max length. """
    if "qwen" in SETTINGS.model_name.lower():
        max_model_len = SETTINGS.context_length if SETTINGS.context_length > 0 else 8192
    else:
        max_model_len = model_config.max_model_len
    return max_model_len


def create_vllm_engine():
    """ get vllm generate engine for chat or completion. """
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.transformers_utils.tokenizer import get_tokenizer
    except ImportError:
        return None

    include = {
        "tokenizer_mode", "trust_remote_code", "tensor_parallel_size",
        "dtype", "gpu_memory_utilization", "max_num_seqs",
    }
    kwargs = SETTINGS.dict(include=include)
    engine_args = AsyncEngineArgs(
        model=SETTINGS.model_path,
        max_num_batched_tokens=SETTINGS.max_num_batched_tokens if SETTINGS.max_num_batched_tokens > 0 else None,
        max_model_len=SETTINGS.context_length if SETTINGS.context_length > 0 else None,
        quantization=SETTINGS.quantization_method,
        **kwargs,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # A separate tokenizer to map token IDs to strings.
    engine.engine.tokenizer = get_tokenizer(
        engine_args.tokenizer,
        tokenizer_mode=engine_args.tokenizer_mode,
        trust_remote_code=True,
    )

    # prompt adapter for constructing model inputs
    engine.prompt_adapter = get_prompt_adapter(
        SETTINGS.model_name.lower(),
        prompt_name=SETTINGS.chat_template.lower() if SETTINGS.chat_template else None
    )

    engine_model_config = asyncio.run(engine.get_model_config())
    engine.engine.scheduler_config.max_model_len = get_context_len(engine_model_config)
    engine.max_model_len = get_context_len(engine_model_config)

    logger.info("Using vllm engine")

    return engine


def create_llama_cpp_engine():
    """ get llama.cpp generate engine for chat or completion. """
    try:
        from llama_cpp import Llama
    except ImportError:
        return None

    include = {
        "n_gpu_layers", "main_gpu", "tensor_split", "n_batch", "n_threads",
        "n_threads_batch", "rope_scaling_type", "rope_freq_base", "rope_freq_scale"
    }
    kwargs = SETTINGS.dict(include=include)
    engine = Llama(
        model_path=SETTINGS.model_path,
        n_ctx=SETTINGS.context_length if SETTINGS.context_length > 0 else 2048,
        **kwargs,
    )

    # prompt adapter for constructing model inputs
    engine.prompt_adapter = get_prompt_adapter(
        SETTINGS.model_name.lower(),
        prompt_name=SETTINGS.chat_template.lower() if SETTINGS.chat_template else None
    )

    logger.info("Using llama.cpp engine")

    return engine


# fastapi app
app = create_app()

# model for embedding
EMBEDDED_MODEL = create_embedding_model() if (SETTINGS.embedding_name and SETTINGS.activate_inference) else None

# model for transformers generate
GENERATE_ENGINE = None
if (not SETTINGS.only_embedding) and SETTINGS.activate_inference:
    if SETTINGS.engine == "default":
        GENERATE_ENGINE = create_generate_model()
    elif SETTINGS.engine == "vllm":
        GENERATE_ENGINE = create_vllm_engine()
    elif SETTINGS.engine == "llama.cpp":
        GENERATE_ENGINE = create_llama_cpp_engine()

# model names for special processing
EXCLUDE_MODELS = ["baichuan-13b", "baichuan2-13b", "qwen", "chatglm3"]
