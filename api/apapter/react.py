import json
from typing import Tuple, List

from api.utils.protocol import (
    ChatFunction,
    Role,
    ChatMessage,
    DeltaMessage,
    FunctionCallResponse,
)

OBSERVATION = "Observation"

TOOL_DESC = """{name}: Call this tool to interact with the {name} API. What is the {name} API useful for? {description} Parameters: {parameters} Format the arguments as a JSON object."""

REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""


def check_function_call(messages: List[ChatMessage], functions: List[ChatFunction] = None):
    """ check need function call or not """
    if functions is not None and len(functions) > 0:
        return True
    if messages is not None and len(messages) > 0 and messages[-1].role == Role.FUNCTION:
        return True
    return False


def build_function_call_messages(messages: List[ChatMessage], functions: List[ChatFunction] = None, function_call="auto"):
    if function_call != "auto" and isinstance(function_call, dict):
        functions = [f for f in functions if f.name in [function_call["name"]]]

    tool_descs, tool_names = [], []
    for f in functions:
        tool_descs.append(
            TOOL_DESC.format(
                name=f.name,
                description=f.description,
                parameters=json.dumps(f.parameters, ensure_ascii=False))
        )
        tool_names.append(f.name)

    tool_descs = "\n\n".join(tool_descs)
    tool_names = ", ".join(tool_names)

    res = ""
    for index, message in enumerate(reversed(messages)):
        role = message.role
        if role == Role.USER:
            res = _build_react_prompt(message, tool_descs, tool_names, OBSERVATION) + res
            break
        elif role == Role.ASSISTANT:
            if message.function_call:
                res = _build_function_call_prompt(message) + res
        elif role == Role.FUNCTION:
            res = _build_function_prompt(message, OBSERVATION) + res

    converted = [ChatMessage(role=Role.USER, content=res)]

    # TODO: filter out the other messages
    for i, message in enumerate(reversed(messages[:-(index + 1)])):
        if message.role == Role.USER or (message.role == Role.ASSISTANT and message.function_call is None):
            converted.append(message)

    return [x for x in reversed(converted)]


def _build_react_prompt(message: ChatMessage, tool_descs: str, tool_names: str, OBSERVATION: str) -> str:
    return REACT_PROMPT.format(
        tool_descs=tool_descs,
        tool_names=tool_names,
        query=message.content,
        OBSERVATION=OBSERVATION,
    )


def _build_function_prompt(message: ChatMessage, OBSERVATION: str) -> str:
    return f"\n{OBSERVATION}: output of {message.name} is {str(message.content).strip()}"


def _build_function_call_prompt(message: ChatMessage) -> str:
    function_name = message.function_call.name
    arguments = message.function_call.arguments
    res = f"\nThought: {message.function_call.thought.strip()}"
    res += f"\nAction: {function_name.strip()}"
    res += f"\nAction Input: {arguments.strip()}"
    return res


def build_chat_message(response: str, functions: List[ChatFunction]) -> Tuple[ChatMessage, str]:
    parsed = _parse_qwen_plugin_call(response)
    if parsed is None:
        return ChatMessage(role="assistant", content=response), "stop"
    else:
        thought, name, args, answer = parsed
        if answer:
            return ChatMessage(role="assistant", content=answer), "stop"
        else:
            function_call = FunctionCallResponse(name=name, arguments=args, thought=thought)
            return ChatMessage(role="assistant", content=None, function_call=function_call, functions=functions), "function_call"


def build_delta_message(text: str, field: str = "name") -> DeltaMessage:
    if field == "arguments":
        return DeltaMessage(function_call=FunctionCallResponse(arguments=text))
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    name = text[i + len('\nAction:'): j].strip()
    return DeltaMessage(function_call=FunctionCallResponse(name=name, arguments=""))


def _parse_qwen_plugin_call(text: str):
    """ parse the generated text """
    t = text.rfind('Thought:')
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')
    l = text.rfind('\nFinal Answer:')

    if l >= 0:
        answer = text[l + len('\nFinal Answer:'):].strip()
        return None, None, None, answer

    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')

    if 0 <= i < j < k:
        thought = text[t + len("Thought:"): i].strip() if t >= 0 else None
        name = text[i + len('\nAction:'): j].strip()
        args = text[j + len('\nAction Input:'): k].strip()
        return thought, name, args, None
    return None
