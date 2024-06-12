from __future__ import annotations

import json
from copy import deepcopy
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    TYPE_CHECKING,
)

from loguru import logger
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

from api.protocol import Role
from api.templates.base import ChatTemplate
from api.templates.registry import register_template

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, BatchEncoding


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

_TEXT_COMPLETION_CMD = object()


def build_qwen_chat_input(
    tokenizer: "PreTrainedTokenizer",
    messages: List[ChatCompletionMessageParam],
    max_window_size: int = 6144,
    functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """
    Builds the input tokens for Qwen chat generation.

    Refs:
        https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py

    Args:
        tokenizer: The tokenizer used to encode the input tokens.
        messages: The list of chat messages.
        max_window_size: The maximum length of the context.
        functions: Optional dictionary or list of dictionaries representing the functions.
        tools: Optional list of dictionaries representing the tools.

    Returns:
        The list of input tokens.
    """
    query, history, system = process_qwen_messages(messages, functions, tools)
    if query is _TEXT_COMPLETION_CMD:
        return build_last_message_input(tokenizer, history, system)

    im_start_tokens, im_end_tokens = [tokenizer.im_start_id], [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    if hasattr(tokenizer, "IMAGE_ST"):
        def _tokenize_str(role, content):
            return tokenizer.encode(
                role, allowed_special=set(tokenizer.IMAGE_ST)
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))
    else:
        def _tokenize_str(role, content):
            return tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

    system_tokens_part = _tokenize_str("system", system)
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

    context_tokens = []
    for turn_query, turn_response in reversed(history):
        query_tokens_part = _tokenize_str("user", turn_query)
        query_tokens = im_start_tokens + query_tokens_part + im_end_tokens

        response_tokens_part = _tokenize_str("assistant", turn_response)
        response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

        next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens

        current_context_size = (
            len(system_tokens) + len(next_context_tokens) + len(context_tokens)
        )
        if current_context_size < max_window_size:
            context_tokens = next_context_tokens + context_tokens
        else:
            break

    context_tokens = system_tokens + context_tokens
    context_tokens += (
        nl_tokens
        + im_start_tokens
        + _tokenize_str("user", query)
        + im_end_tokens
        + nl_tokens
        + im_start_tokens
        + tokenizer.encode("assistant")
        + nl_tokens
    )

    return context_tokens


def process_qwen_messages(
    messages: List[ChatCompletionMessageParam],
    functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[List[str]], str]:
    """
    Process the Qwen messages and generate a query and history.

    Args:
        messages (List[ChatCompletionMessageParam]): The list of chat completion messages.
        functions (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]): The functions to be used.
        tools (Optional[List[Dict[str, Any]]]): The tools to be used.

    Returns:
        Tuple[str, List[List[str]], str]: The generated query and history and system.
    """
    if all(m["role"] != Role.USER.value for m in messages):
        raise ValueError(f"Invalid messages: Expecting at least one user message.")

    messages = deepcopy(messages)
    if messages[0]["role"] == Role.SYSTEM.value:
        system = messages.pop(0)["content"].lstrip("\n").rstrip()
    else:
        system = "You are a helpful assistant."

    if tools:
        functions = [t["function"] for t in tools]

    if functions:
        tools_text = []
        tools_name_text = []
        for func_info in functions:
            name = func_info.get("name", "")
            name_m = func_info.get("name_for_model", name)
            name_h = func_info.get("name_for_human", name)
            desc = func_info.get("description", "")
            desc_m = func_info.get("description_for_model", desc)
            tool = TOOL_DESC.format(
                name_for_model=name_m,
                name_for_human=name_h,
                # Hint: You can add the following format requirements in description:
                #   "Format the arguments as a JSON object."
                #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                description_for_model=desc_m,
                parameters=json.dumps(func_info["parameters"], ensure_ascii=False),
            )

            tools_text.append(tool)
            tools_name_text.append(name_m)

        tools_text = "\n\n".join(tools_text)
        tools_name_text = ", ".join(tools_name_text)
        instruction = REACT_INSTRUCTION.format(
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        ).lstrip('\n').rstrip()
    else:
        instruction = ""

    messages_with_fncall = messages
    messages = []
    for m_idx, m in enumerate(messages_with_fncall):
        role, content = m["role"], m["content"]
        func_call, tool_calls = m.get("function_call", None), m.get("tool_calls", None)

        content = content or ''
        content = content.lstrip('\n').rstrip()

        if role in [Role.FUNCTION.value, Role.TOOL.value]:
            if (len(messages) == 0) or (messages[-1]["role"] != Role.ASSISTANT.value):
                raise ValueError(f"Invalid messages: Expecting role assistant before role function.")

            messages[-1]["content"] += f"\nObservation: {content}"
            if m_idx == len(messages_with_fncall) - 1:
                messages[-1]["content"] += "\nThought:"

        elif role == Role.ASSISTANT.value:
            if len(messages) == 0:
                raise ValueError(f"Invalid messages: Expecting role user before role assistant.")

            if func_call is None and tool_calls is None:
                if functions or tool_calls:
                    content = f"Thought: I now know the final answer.\nFinal Answer: {content}"

            if messages[-1]["role"] in [Role.USER.value, Role.SYSTEM.value]:
                messages.append(
                    ChatCompletionAssistantMessageParam(role="assistant", content=content.lstrip("\n").rstrip())
                )
            else:
                messages[-1]["content"] += content
        elif role in [Role.USER.value, Role.SYSTEM.value]:
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=content.lstrip("\n").rstrip())
            )

    query = _TEXT_COMPLETION_CMD
    if messages[-1]["role"] == Role.USER.value:
        query = messages[-1]["content"]
        messages = messages[:-1]

    if len(messages) % 2 != 0:
        raise ValueError("Invalid messages")

    history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
    for i in range(0, len(messages), 2):
        if messages[i]["role"] == Role.USER.value and messages[i + 1]["role"] == Role.ASSISTANT.value:
            usr_msg = messages[i]["content"].lstrip("\n").rstrip()
            bot_msg = messages[i + 1]["content"].lstrip("\n").rstrip()
            if instruction and (i == len(messages) - 2):
                usr_msg = f"{instruction}\n\nQuestion: {usr_msg}"
                instruction = ""
            history.append([usr_msg, bot_msg])
        else:
            raise ValueError("Invalid messages: Expecting exactly one user (or function) role before every assistant role.")

    if instruction:
        assert query is not _TEXT_COMPLETION_CMD
        query = f"{instruction}\n\nQuestion: {query}"

    return query, history, system


def build_last_message_input(tokenizer: "PreTrainedTokenizer", history: List[List[str]], system: str):
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    prompt = f"{im_start}system\n{system}{im_end}"
    for i, (query, response) in enumerate(history):
        query = query.lstrip("\n").rstrip()
        response = response.lstrip("\n").rstrip()
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{response}{im_end}"
    prompt = prompt[:-len(im_end)]
    logger.debug(f"==== Prompt with tools ====\n{prompt}")
    return tokenizer.encode(prompt)


@register_template("qwen")
class QwenChatTemplate(ChatTemplate):
    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    stop = ["<|endoftext|>", "<|im_end|>"]
    stop_token_ids = [151643, 151644, 151645]  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
    function_call_available = True

    def _convert_messages_to_ids(
        self,
        messages: List[ChatCompletionMessageParam],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = 256,
        max_window_size: Optional[int] = 6144,
        **kwargs,
    ) -> Union[List[int], BatchEncoding]:
        return build_qwen_chat_input(
            self.tokenizer,
            messages,
            max_window_size,
            tools=tools,
        )

    def parse_assistant_response(
        self,
        output: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[Union[str, Dict[str, Any]]]]:
        func_name, func_args = "", ""
        i = output.rfind("\nAction:")
        j = output.rfind("\nAction Input:")
        k = output.rfind("\nObservation:")

        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                # then it is likely that `Observation` is omitted by the LLM,
                # because the output text may have discarded the stop word.
                output = output.rstrip() + "\nObservation:"  # Add it back.
            k = output.rfind("\nObservation:")
            func_name = output[i + len("\nAction:"): j].strip()
            func_args = output[j + len("\nAction Input:"): k].strip()

        if func_name:
            function_call = {
                "function": {
                    "name": func_name,
                    "arguments": func_args
                },
                "id": func_name,
                "type": "function",
            }
            return output[:k], function_call

        z = output.rfind("\nFinal Answer: ")
        if z >= 0:
            output = output[z + len("\nFinal Answer: "):]
        return output, None

    @property
    def chat_template(self) -> str:
        """ This template formats inputs in the standard ChatML format. See
        https://github.com/openai/openai-python/blob/main/chatml.md
        """
        return (
            "{{ system_prompt }}"
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )


@register_template("qwen2")
class Qwen2ChatTemplate(ChatTemplate):
    stop = ["<|endoftext|>", "<|im_end|>"]
