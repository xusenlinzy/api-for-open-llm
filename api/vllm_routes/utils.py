from http import HTTPStatus

from fastapi.responses import JSONResponse

from api.generation import build_qwen_chat_input, build_baichuan_chat_input
from api.models import EXCLUDE_MODELS, VLLM_ENGINE
from api.utils.protocol import ErrorResponse


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").dict(), status_code=status_code.value
    )


async def get_gen_prompt(request, model_name):
    if any(m in model_name for m in EXCLUDE_MODELS):
        return request.messages
    else:
        return VLLM_ENGINE.prompt_adapter.generate_prompt(request.messages)


async def get_model_inputs(request, prompt, model_name):
    if isinstance(prompt, str):
        input_ids = VLLM_ENGINE.encode_tokenizer(prompt).input_ids
    elif isinstance(prompt[0], int):
        input_ids = prompt
    else:
        if "baichuan-13b" in model_name:
            input_ids = build_baichuan_chat_input(VLLM_ENGINE.encode_tokenizer, prompt)
        elif "qwen" in model_name:
            input_ids = build_qwen_chat_input(VLLM_ENGINE.encode_tokenizer, prompt)
        else:
            raise ValueError(f"Model not supported yet: {model_name}")

    token_num = len(input_ids)
    if token_num + request.max_tokens > VLLM_ENGINE.max_model_len:
        return input_ids, create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {VLLM_ENGINE.max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return input_ids, None
