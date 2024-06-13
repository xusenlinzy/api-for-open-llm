from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.common import dictify
from api.config import SETTINGS


def create_app() -> FastAPI:
    import gc
    import torch

    def torch_gc() -> None:
        r"""
        Collects GPU memory.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @asynccontextmanager
    async def lifespan(app: "FastAPI"):  # collects GPU memory
        yield
        torch_gc()

    """ create fastapi app server """
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def create_rag_models():
    """ get rag models. """
    rag_models = []
    if "rag" in SETTINGS.tasks and SETTINGS.activate_inference:
        if SETTINGS.embedding_name:
            from api.rag import RAGEmbedding
            rag_models.append(
               RAGEmbedding(SETTINGS.embedding_name, SETTINGS.embedding_device)
            )
        else:
            rag_models.append(None)
        if SETTINGS.rerank_name:
            from api.rag import RAGReranker
            rag_models.append(
                RAGReranker(SETTINGS.rerank_name, device=SETTINGS.rerank_device)
            )
        else:
            rag_models.append(None)
    return rag_models if len(rag_models) == 2 else [None, None]


def create_hf_llm():
    """ get generate model for chat or completion. """
    from api.engine.hf import HuggingFaceEngine
    from api.adapter.loader import load_model_and_tokenizer

    include = {
        "device_map",
        "load_in_8bit",
        "load_in_4bit",
        "dtype",
        "rope_scaling",
        "flash_attn",
    }
    kwargs = dictify(SETTINGS, include=include)

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=SETTINGS.model_path, **kwargs,
    )

    logger.info("Using HuggingFace Engine")

    return HuggingFaceEngine(
        model,
        tokenizer,
        model_name=SETTINGS.model_name,
        max_model_length=SETTINGS.context_length if SETTINGS.context_length > 0 else None,
        template_name=SETTINGS.chat_template,
    )


def create_vllm_engine():
    """ get vllm generate engine for chat or completion. """
    try:
        import vllm
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from api.engine.vllm_engine import VllmEngine
    except ImportError:
        raise ValueError("VLLM engine not available")

    vllm_version = vllm.__version__

    include = {
        "tokenizer_mode",
        "trust_remote_code",
        "tensor_parallel_size",
        "dtype",
        "gpu_memory_utilization",
        "max_num_seqs",
        "enforce_eager",
        "lora_extra_vocab_size",
        "disable_custom_all_reduce",
    }

    if vllm_version >= "0.4.3":
        include.add("max_seq_len_to_capture")
        include.add("distributed_executor_backend")

    kwargs = dictify(SETTINGS, include=include)
    engine_args = AsyncEngineArgs(
        model=SETTINGS.model_path,
        max_num_batched_tokens=SETTINGS.max_num_batched_tokens if SETTINGS.max_num_batched_tokens > 0 else None,
        max_model_len=SETTINGS.context_length if SETTINGS.context_length > 0 else None,
        quantization=SETTINGS.quantization_method,
        max_cpu_loras=SETTINGS.max_cpu_loras if SETTINGS.max_cpu_loras > 0 else None,
        disable_log_stats=SETTINGS.vllm_disable_log_stats,
        disable_log_requests=True,
        **kwargs,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    logger.info("Using vllm engine")

    return VllmEngine(
        engine,
        SETTINGS.model_name,
        SETTINGS.chat_template,
    )


# fastapi app
app = create_app()

# model for rag
EMBEDDING_MODEL, RERANK_MODEL = create_rag_models()

# llm
if "llm" in SETTINGS.tasks and SETTINGS.activate_inference:
    if SETTINGS.engine == "default":
        LLM_ENGINE = create_hf_llm()
    elif SETTINGS.engine == "vllm":
        LLM_ENGINE = create_vllm_engine()
else:
    LLM_ENGINE = None
