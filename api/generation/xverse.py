from typing import List

from openai.types.chat import ChatCompletionMessageParam
from transformers import PreTrainedTokenizer

from api.generation.utils import parse_messages
from api.utils.protocol import Role


def build_xverse_chat_input(
    tokenizer: PreTrainedTokenizer,
    messages: List[ChatCompletionMessageParam],
    context_len: int = 8192,
    max_new_tokens: int = 256
) -> List[int]:
    """
    Builds the input tokens for the Xverse chat model based on the given messages.

    Refs:
        https://huggingface.co/xverse/XVERSE-13B-Chat/blob/main/modeling_xverse.py

    Args:
        tokenizer: The PreTrainedTokenizer object.
        messages: A list of ChatCompletionMessageParam objects representing the chat messages.
        context_len: The maximum length of the context (default=8192).
        max_new_tokens: The maximum number of new tokens to be added (default=256).

    Returns:
        List[int]: The input tokens for the Baichuan chat model.
    """
    max_input_tokens = context_len - max_new_tokens
    system, rounds = parse_messages(messages)
    system = f"{system}\n\n" if system else system

    def _tokenize_str(role, content):
        return tokenizer.encode(f"{role}: {content}", return_token_type_ids=False)

    system_tokens = tokenizer.encode(system, return_token_type_ids=False)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for i, r in enumerate(rounds[::-1]):
        round_tokens = []
        for message in r:
            if message["role"] == Role.USER:
                content = f"{message['content']}\n\n"
                if i == 0:
                    content += "Assistant: "
                content_tokens = _tokenize_str("Human", content)
            else:
                content_tokens = _tokenize_str("Assistant", f"{message['content']}") + [3]  # add eos token id

            round_tokens.extend(content_tokens)

        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    return input_tokens[-max_input_tokens:]  # truncate left


def check_is_xverse(model) -> bool:
    """
    Checks if the given model is a Xverse model.

    Args:
        model: The model to be checked.

    Returns:
        bool: True if the model is a Xverse model, False otherwise.
    """
    return "XverseDecoderLayer" in getattr(model, "_no_split_modules", [])
