import json


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
    if function_call != "auto":
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
            ret += f"\n{content.strip()}"
        elif role == "function":
            ret += f"\nObservation: {content.strip()}"

    return [{"role": "user", "content": ret}] if return_messages else ret
