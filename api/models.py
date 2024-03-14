from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.config import SETTINGS
from api.utils.compat import model_dump


def create_app() -> FastAPI:
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
    if SETTINGS.tei_endpoint is not None:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=SETTINGS.tei_endpoint, api_key="none")
    else:
        from sentence_transformers import SentenceTransformer
        client = SentenceTransformer(SETTINGS.embedding_name, device=SETTINGS.embedding_device)
    return client


def create_generate_model():
    """ get generate model for chat or completion. """
    from api.core.default import DefaultEngine
    from api.adapter.loader import load_model_and_tokenizer

    include = {
        "model_name",
        "quantize",
        "device",
        "device_map",
        "num_gpus",
        "pre_seq_len",
        "load_in_8bit",
        "load_in_4bit",
        "using_ptuning_v2",
        "dtype",
        "resize_embeddings",
        "rope_scaling",
        "flash_attn",
    }
    kwargs = model_dump(SETTINGS, include=include)

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=SETTINGS.model_path,
        adapter_model=SETTINGS.adapter_model_path,
        **kwargs,
    )

    logger.info("Using default engine")

    return DefaultEngine(
        model,
        tokenizer,
        SETTINGS.device,
        model_name=SETTINGS.model_name,
        context_len=SETTINGS.context_length if SETTINGS.context_length > 0 else None,
        prompt_name=SETTINGS.chat_template,
        use_streamer_v2=SETTINGS.use_streamer_v2,
    )


def create_vllm_engine():
    """ get vllm generate engine for chat or completion. """
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from api.core.vllm_engine import VllmEngine, LoRA
    except ImportError:
        return None

    include = {
        "tokenizer_mode",
        "trust_remote_code",
        "tensor_parallel_size",
        "dtype",
        "gpu_memory_utilization",
        "max_num_seqs",
        "enforce_eager",
        "max_context_len_to_capture",
        "max_loras",
        "max_lora_rank",
        "lora_extra_vocab_size",
    }
    kwargs = model_dump(SETTINGS, include=include)
    engine_args = AsyncEngineArgs(
        model=SETTINGS.model_path,
        max_num_batched_tokens=SETTINGS.max_num_batched_tokens if SETTINGS.max_num_batched_tokens > 0 else None,
        max_model_len=SETTINGS.context_length if SETTINGS.context_length > 0 else None,
        quantization=SETTINGS.quantization_method,
        max_cpu_loras=SETTINGS.max_cpu_loras if SETTINGS.max_cpu_loras > 0 else None,
        **kwargs,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    logger.info("Using vllm engine")

    lora_modules = []
    for item in SETTINGS.lora_modules.strip().split("+"):
        if "=" in item:
            name, path = item.split("=")
            lora_modules.append(LoRA(name, path))

    return VllmEngine(
        engine,
        SETTINGS.model_name,
        SETTINGS.chat_template,
        lora_modules=lora_modules,
    )


def create_llama_cpp_engine():
    """ get llama.cpp generate engine for chat or completion. """
    try:
        from llama_cpp import Llama
        from api.core.llama_cpp_engine import LlamaCppEngine
    except ImportError:
        return None

    include = {
        "n_gpu_layers",
        "main_gpu",
        "tensor_split",
        "n_batch",
        "n_threads",
        "n_threads_batch",
        "rope_scaling_type",
        "rope_freq_base",
        "rope_freq_scale",
    }
    kwargs = model_dump(SETTINGS, include=include)
    engine = Llama(
        model_path=SETTINGS.model_path,
        n_ctx=SETTINGS.context_length if SETTINGS.context_length > 0 else 2048,
        **kwargs,
    )

    logger.info("Using llama.cpp engine")

    return LlamaCppEngine(engine, SETTINGS.model_name, SETTINGS.chat_template)


def create_tgi_engine():
    """ get llama.cpp generate engine for chat or completion. """
    try:
        from text_generation import AsyncClient
        from api.core.tgi import TGIEngine
    except ImportError:
        return None

    client = AsyncClient(SETTINGS.tgi_endpoint)
    logger.info("Using TGI engine")

    return TGIEngine(client, SETTINGS.model_name, SETTINGS.chat_template)


# fastapi app
app = create_app()

# model for embedding
EMBEDDED_MODEL = create_embedding_model() if (SETTINGS.embedding_name and SETTINGS.activate_inference) else None

# model for transformers generate
if (not SETTINGS.only_embedding) and SETTINGS.activate_inference:
    if SETTINGS.engine == "default":
        GENERATE_ENGINE = create_generate_model()
    elif SETTINGS.engine == "vllm":
        GENERATE_ENGINE = create_vllm_engine()
    elif SETTINGS.engine == "llama.cpp":
        GENERATE_ENGINE = create_llama_cpp_engine()
    elif SETTINGS.engine == "tgi":
        GENERATE_ENGINE = create_tgi_engine()
else:
    GENERATE_ENGINE = None

# model names for special processing
EXCLUDE_MODELS = ["baichuan-13b", "baichuan2-13b", "qwen", "chatglm3"]
