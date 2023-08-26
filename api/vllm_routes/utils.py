from http import HTTPStatus

from fastapi.responses import JSONResponse

from api.config import config
from api.generation import build_qwen_chat_input, build_baichuan_chat_input
from api.models import EXCLUDE_MODELS, VLLM_ENGINE
from api.utils.protocol import ErrorResponse


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").dict(), status_code=status_code.value
    )


async def get_gen_prompt(request, model_name):
    if any(m in model_name for m in EXCLUDE_MODELS) and config.PROMPT_NAME is None:
        return request.messages
    else:
        return VLLM_ENGINE.prompt_adapter.generate_prompt(request.messages)


async def get_model_inputs(request, prompt, model_name):
    max_input_tokens = VLLM_ENGINE.max_model_len - request.max_tokens
    if isinstance(prompt, str):
        if getattr(request, "infilling", False):
            input_ids = VLLM_ENGINE.engine.tokenizer(
                prompt,
                suffix_first=getattr(request, "suffix_first", False)
            ).input_ids
        else:
            input_ids = VLLM_ENGINE.engine.tokenizer(prompt).input_ids[-max_input_tokens:]  # truncate left
    elif isinstance(prompt[0], int):
        input_ids = prompt[-max_input_tokens:]  # truncate left
    else:
        if "baichuan-13b" in model_name:
            input_ids = build_baichuan_chat_input(
                VLLM_ENGINE.engine.tokenizer,
                prompt,
                max_new_tokens=request.max_tokens,
            )
        elif "qwen" in model_name:
            input_ids = build_qwen_chat_input(
                VLLM_ENGINE.engine.tokenizer,
                prompt,
                max_new_tokens=request.max_tokens,
            )
        else:
            raise ValueError(f"Model not supported yet: {model_name}")
    return input_ids, None
