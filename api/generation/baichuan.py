from typing import List

from api.utils.protocol import Role, ChatMessage


def build_baichuan_chat_input(tokenizer, messages: List[ChatMessage], context_len: int = 4096):
    """  https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/modeling_baichuan.py """
    total_input, round_input = [], []
    for message in messages[::-1]:
        role, content_tokens = message.role, tokenizer.encode(message.content)
        if role in [Role.USER, Role.SYSTEM]:
            round_input = [195] + content_tokens + round_input
            if total_input and len(total_input) + len(round_input) > context_len:
                break
            else:
                total_input = round_input + total_input
                round_input = []
        elif role == Role.ASSISTANT:
            round_input = [196] + content_tokens + round_input
        else:
            raise ValueError(f"message role not supported yet: {role}")
    total_input = total_input[-context_len:]  # truncate left
    total_input.append(196)
    return total_input


def check_is_baichuan(model):
    return "BaichuanLayer" in getattr(model, "_no_split_modules", [])
