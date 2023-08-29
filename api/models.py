import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from sentence_transformers import SentenceTransformer

from api.apapter import get_prompt_adapter
from api.config import config


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


def create_embedding_model() -> SentenceTransformer:
    """ get embedding model from sentence-transformers. """
    return SentenceTransformer(config.EMBEDDING_NAME, device=config.EMBEDDING_DEVICE)


def create_generate_model():
    """ get generate model for chat or completion. """
    from api.generation import ModelServer
    from api.apapter.model import load_model

    if config.PATCH_TYPE == "attention":
        from api.utils.patches import apply_attention_patch

        apply_attention_patch(use_memory_efficient_attention=True)
    if config.PATCH_TYPE == "ntk":
        from api.utils.patches import apply_ntk_scaling_patch

        apply_ntk_scaling_patch(config.ALPHA)

    model, tokenizer = load_model(
        config.MODEL_NAME,
        model_name_or_path=config.MODEL_PATH,
        adapter_model=config.ADAPTER_MODEL_PATH,
        quantize=config.QUANTIZE,
        device=config.DEVICE,
        device_map=config.DEVICE_MAP,
        num_gpus=config.NUM_GPUs,
        load_in_8bit=config.LOAD_IN_8BIT,
        load_in_4bit=config.LOAD_IN_4BIT,
        use_ptuning_v2=config.USING_PTUNING_V2,
    )

    return ModelServer(
        model,
        tokenizer,
        config.DEVICE,
        model_name=config.MODEL_NAME,
        context_len=config.CONTEXT_LEN,
        stream_interval=config.STREAM_INTERVERL,
        prompt_name=config.PROMPT_NAME,
        use_streamer_v2=config.USE_STREAMER_V2,
    )


def get_context_len(model_config) -> int:
    """ fix for model max length. """
    if "qwen" in config.MODEL_NAME.lower():
        max_model_len = config.CONTEXT_LEN or 8192
    else:
        max_model_len = config.CONTEXT_LEN or model_config.get_max_model_len()
    return max_model_len


def create_vllm_engine():
    """ get vllm generate engine for chat or completion. """
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.transformers_utils.tokenizer import get_tokenizer
    except ImportError:
        return None

    engine_args = AsyncEngineArgs(
        model=config.MODEL_PATH,
        tokenizer_mode=config.TOKENIZE_MODE,
        trust_remote_code=config.TRUST_REMOTE_CODE,
        dtype=config.DTYPE,
        tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        max_num_batched_tokens=config.MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=config.MAX_NUM_SEQS,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # A separate tokenizer to map token IDs to strings.
    if "code-llama" in config.MODEL_NAME.lower():
        try:
            from transformers import CodeLlamaTokenizer

            engine.engine.tokenizer = CodeLlamaTokenizer.from_pretrained(engine_args.tokenizer)
        except ImportError:
            logger.error(
                "transformers is not installed correctly. Please use the following command to install transformers\npip install git+https://github.com/huggingface/transformers.git."
            )
    else:
        engine.engine.tokenizer = get_tokenizer(
            engine_args.tokenizer,
            tokenizer_mode=engine_args.tokenizer_mode,
            trust_remote_code=True,
        )

    # prompt adapter for constructing model inputs
    engine.prompt_adapter = get_prompt_adapter(
        config.MODEL_NAME.lower(),
        prompt_name=config.PROMPT_NAME.lower() if config.PROMPT_NAME else None
    )

    engine_model_config = asyncio.run(engine.get_model_config())
    engine.engine.scheduler_config.max_model_len = get_context_len(engine_model_config)
    engine.max_model_len = get_context_len(engine_model_config)

    return engine


# fastapi app
app = create_app()

# model for embedding
EMBEDDED_MODEL = create_embedding_model() if (config.EMBEDDING_NAME and config.ACTIVATE_INFERENCE) else None

# model for transformers generate
GENERATE_MDDEL = create_generate_model() if (not config.USE_VLLM and config.ACTIVATE_INFERENCE) else None

# model for vllm generate
VLLM_ENGINE = create_vllm_engine() if (config.USE_VLLM and config.ACTIVATE_INFERENCE) else None

# model names for special processing
EXCLUDE_MODELS = ["baichuan-13b", "qwen"]
