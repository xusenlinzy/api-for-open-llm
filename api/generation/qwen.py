from typing import List

from transformers import PreTrainedTokenizer

from api.utils.protocol import Role, ChatMessage


def build_qwen_chat_input(
    tokenizer: PreTrainedTokenizer,
    messages: List[ChatMessage],
    max_window_size: int = 6144,
):
    """ https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py """
    im_start_tokens, im_end_tokens = [tokenizer.im_start_id], [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    def _tokenize_str(role, content):
        return tokenizer.encode(
            role, allowed_special=set()
        ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

    system_tokens_part = _tokenize_str("system", "You are a helpful assistant.")
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

    context_tokens = []
    for i, message in enumerate(messages[::-1]):
        role, content = message.role, message.content
        if context_tokens:
            context_tokens = nl_tokens + context_tokens

        if role == Role.USER:
            content_tokens = _tokenize_str("user", content)
        elif role == Role.SYSTEM:
            content_tokens = _tokenize_str("system", content)
        elif role == Role.ASSISTANT:
            content_tokens = _tokenize_str("assistant", content)
        else:
            raise ValueError(f"message role not supported yet: {role}")

        if len(im_start_tokens + content_tokens + im_end_tokens + context_tokens) > max_window_size:
            break
        else:
            context_tokens = im_start_tokens + content_tokens + im_end_tokens + context_tokens

    context_tokens = system_tokens + nl_tokens + context_tokens
    return context_tokens + nl_tokens + im_start_tokens + tokenizer.encode("assistant") + nl_tokens


def check_is_qwen(model):
    return "QWenBlock" in getattr(model, "_no_split_modules", [])
