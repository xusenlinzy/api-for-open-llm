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
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SETTINGS.embedding_name, device=SETTINGS.embedding_device)


def create_generate_model():
    """ get generate model for chat or completion. """
    from api.core.default import DefaultEngine
    from api.adapter.model import load_model

    if SETTINGS.patch_type == "attention":
        from api.utils.patches import apply_attention_patch

        apply_attention_patch(use_memory_efficient_attention=True)
    if SETTINGS.patch_type == "ntk":
        from api.utils.patches import apply_ntk_scaling_patch

        apply_ntk_scaling_patch(SETTINGS.alpha)

    include = {
        "model_name", "quantize", "device", "device_map", "num_gpus", "pre_seq_len",
        "load_in_8bit", "load_in_4bit", "using_ptuning_v2", "dtype", "resize_embeddings"
    }
    kwargs = model_dump(SETTINGS, include=include)

    model, tokenizer = load_model(
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
        from vllm.transformers_utils.tokenizer import get_tokenizer
        from api.core.vllm_engine import VllmEngine
    except ImportError:
        return None

    include = {
        "tokenizer_mode", "trust_remote_code", "tensor_parallel_size",
        "dtype", "gpu_memory_utilization", "max_num_seqs",
    }
    kwargs = model_dump(SETTINGS, include=include)
    engine_args = AsyncEngineArgs(
        model=SETTINGS.model_path,
        max_num_batched_tokens=SETTINGS.max_num_batched_tokens if SETTINGS.max_num_batched_tokens > 0 else None,
        max_model_len=SETTINGS.context_length if SETTINGS.context_length > 0 else None,
        quantization=SETTINGS.quantization_method,
        **kwargs,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_args.tokenizer,
        tokenizer_mode=engine_args.tokenizer_mode,
        trust_remote_code=True,
    )

    logger.info("Using vllm engine")

    return VllmEngine(
        engine,
        tokenizer,
        SETTINGS.model_name,
        SETTINGS.chat_template,
        SETTINGS.context_length,
    )


def create_llama_cpp_engine():
    """ get llama.cpp generate engine for chat or completion. """
    try:
        from llama_cpp import Llama
        from api.core.llama_cpp_engine import LlamaCppEngine
    except ImportError:
        return None

    include = {
        "n_gpu_layers", "main_gpu", "tensor_split", "n_batch", "n_threads",
        "n_threads_batch", "rope_scaling_type", "rope_freq_base", "rope_freq_scale"
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
