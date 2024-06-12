from __future__ import annotations

from typing import List, Tuple

from openai.types.chat import ChatCompletionMessageParam
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from api.protocol import Role


def parse_messages(
    messages: List[ChatCompletionMessageParam], split_role=Role.USER.value
) -> Tuple[str, List[List[ChatCompletionMessageParam]]]:
    """
    Parse a list of chat completion messages into system and rounds.

    Args:
        messages (List[ChatCompletionMessageParam]): The list of chat completion messages.
        split_role: The role at which to split the rounds. Defaults to Role.USER.

    Returns:
        Tuple[str, List[List[ChatCompletionMessageParam]]]: A tuple containing the system message and a list of rounds.
    """
    system, rounds = "", []
    r = []
    for i, message in enumerate(messages):
        if message["role"] == Role.SYSTEM.value:
            system = message["content"]
            continue
        if message["role"] == split_role and r:
            rounds.append(r)
            r = []
        r.append(message)
    if r:
        rounds.append(r)
    return system, rounds


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    """
    Prepare a list of logits processors based on the provided parameters.

    Args:
        temperature (float): The temperature value for temperature warping.
        repetition_penalty (float): The repetition penalty value.
        top_p (float): The top-p value for top-p warping.
        top_k (int): The top-k value for top-k warping.

    Returns:
        LogitsProcessorList: A list of logits processors.
    """
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op, so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def is_partial_stop(output: str, stop_str: str):
    """ Check whether the output contains a partial stop str. """
    return any(
        stop_str.startswith(output[-i:])
        for i in range(0, min(len(output), len(stop_str)))
    )


# Models don't use the same configuration key for determining the maximum
# sequence length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important.  Some models have two of these, and we
# have a preference for which value gets used.
SEQUENCE_LENGTH_KEYS = [
    "max_position_embeddings",
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
]


def get_context_length(config) -> int:
    """ Get the context length of a model from a huggingface model config. """
    rope_scaling = getattr(config, "rope_scaling", None)
    rope_scaling_factor = config.rope_scaling["factor"] if rope_scaling else 1
    for key in SEQUENCE_LENGTH_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


def apply_stopping_strings(reply: str, stop_strings: List[str]) -> Tuple[str, bool]:
    """
    Apply stopping strings to the reply and check if a stop string is found.

    Args:
        reply (str): The reply to apply stopping strings to.
        stop_strings (List[str]): The list of stopping strings to check for.

    Returns:
        Tuple[str, bool]: A tuple containing the modified reply and a boolean indicating if a stop string was found.
    """
    stop_found = False
    for string in stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou: is completed, trim it
        for string in stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found
