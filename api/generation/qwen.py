from typing import List

from transformers import PreTrainedTokenizer

from api.generation.utils import parse_messages
from api.utils.protocol import Role, ChatMessage


def build_qwen_chat_input(
    tokenizer: PreTrainedTokenizer,
    messages: List[ChatMessage],
    context_len: int = 8192,
    max_new_tokens: int = 256
) -> List[int]:
    """ https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py """
    max_input_tokens = context_len - max_new_tokens
    system, rounds = parse_messages(messages)
    system = "You are a helpful assistant." + system  # fix system prompt

    im_start_tokens, im_end_tokens = [tokenizer.im_start_id], [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    def _tokenize_str(role, content):
        return tokenizer.encode(
            role, allowed_special=set()
        ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

    system_tokens_part = _tokenize_str("system", system)
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for r in rounds[::-1]:
        round_tokens = []
        for message in r:
            if round_tokens:
                round_tokens += nl_tokens

            if message.role == Role.USER:
                content_tokens = im_start_tokens + _tokenize_str("user", message.content) + im_end_tokens
            else:
                content_tokens = im_start_tokens + _tokenize_str("assistant", message.content) + im_end_tokens

            round_tokens.extend(content_tokens)

        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            if history_tokens:
                history_tokens = nl_tokens + history_tokens

            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + nl_tokens + history_tokens
    if messages[-1].role != Role.ASSISTANT:
        input_tokens += nl_tokens + im_start_tokens + tokenizer.encode("assistant") + nl_tokens
    return input_tokens[-max_input_tokens:]  # truncate left


def check_is_qwen(model) -> bool:
    return "QWenBlock" in getattr(model, "_no_split_modules", [])
