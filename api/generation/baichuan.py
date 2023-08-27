from typing import List

from transformers import PreTrainedTokenizer

from api.generation.utils import parse_messages
from api.utils.protocol import Role, ChatMessage


def build_baichuan_chat_input(
    tokenizer: PreTrainedTokenizer,
    messages: List[ChatMessage],
    context_len: int = 4096,
    max_new_tokens: int = 256
) -> List[int]:
    """  https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_utils.py """
    max_input_tokens = context_len - max_new_tokens
    system, rounds = parse_messages(messages)
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for r in rounds[::-1]:
        round_tokens = []
        for message in r:
            if message.role == Role.USER:
                round_tokens.append(195)
            else:
                round_tokens.append(196)
            round_tokens.extend(tokenizer.encode(message.content))

        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1].role != Role.ASSISTANT:
        input_tokens.append(196)

    return input_tokens[-max_input_tokens:]  # truncate left


def check_is_baichuan(model) -> bool:
    return "BaichuanLayer" in getattr(model, "_no_split_modules", [])
