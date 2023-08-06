import json
from typing import Tuple, Union


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

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


def get_qwen_react_prompt(messages, functions, function_call="auto", return_messages=True):
    if function_call != "auto" and isinstance(function_call, dict):
        functions = [info for info in functions if info["name_for_model"] in [function_call["name_for_model"]]]

    tool_descs, tool_names = [], []
    for info in functions:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model=info["name_for_model"],
                name_for_human=info["name_for_human"],
                description_for_model=info["description_for_model"],
                parameters=json.dumps(info["parameters"], ensure_ascii=False))
        )
        tool_names.append(info["name_for_model"])

    tool_descs = "\n\n".join(tool_descs)
    tool_names = ",".join(tool_names)

    ret = ""
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "user":
            ret += REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, query=content)
        elif role == "assistant":
            if message.get("function_call"):
                thought = message["function_call"]["thought"]
                function_name = message["function_call"]["name"]
                arguments = message["function_call"]["arguments"]

                if thought is not None:
                    ret += f"\nThought: {thought.strip()}"

                ret += f"\nAction: {function_name.strip()}"
                ret += f"\nAction Input: {arguments.strip()}"
        elif role == "function":
            ret += f"\nObservation: output of {message['name']} is {str(content).strip()}"

    return [{"role": "user", "content": ret}] if return_messages else ret


def parse_qwen_plugin_call(text: str) -> Union[Tuple[str, str, str], None]:
    t = text.rfind('Thought:')
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')

    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')

    if 0 <= i < j < k:
        thought = text[t + len("Thought:"): i].strip() if t >= 0 else None
        plugin_name = text[i + len('\nAction:'): j].strip()
        plugin_args = text[j + len('\nAction Input:'): k].strip()
        return thought, plugin_name, plugin_args
    return None
