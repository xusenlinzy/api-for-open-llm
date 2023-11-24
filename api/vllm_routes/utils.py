from api.config import SETTINGS
from api.generation import build_qwen_chat_input, build_baichuan_chat_input
from api.models import EXCLUDE_MODELS


async def get_gen_prompt(engine, request, model_name):
    if any(m in model_name for m in EXCLUDE_MODELS) and SETTINGS.chat_template is None:
        return request.messages
    else:
        return engine.prompt_adapter.apply_chat_template(request.messages)


async def get_model_inputs(engine, request, prompt, model_name):
    max_input_tokens = engine.max_model_len - request.max_tokens
    if isinstance(prompt, str):
        if getattr(request, "infilling", False):
            input_ids = engine.engine.tokenizer(
                prompt,
                suffix_first=getattr(request, "suffix_first", False)
            ).input_ids
        else:
            input_ids = engine.engine.tokenizer(prompt).input_ids[-max_input_tokens:]  # truncate left
    elif isinstance(prompt[0], int):
        input_ids = prompt[-max_input_tokens:]  # truncate left
    else:
        if ("baichuan-13b" in model_name) or ("baichuan2-13b" in model_name):
            input_ids = build_baichuan_chat_input(
                engine.engine.tokenizer,
                prompt,
                max_new_tokens=request.max_tokens,
            )
        elif "qwen" in model_name:
            input_ids = build_qwen_chat_input(
                engine.engine.tokenizer,
                prompt,
                max_new_tokens=request.max_tokens,
                functions=request.functions,
            )
        else:
            raise ValueError(f"Model not supported yet: {model_name}")
    return input_ids, None
