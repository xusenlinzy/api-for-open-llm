""" Adapted from https://github.com/InternLM/lagent/blob/main/lagent/agents/react.py """

import json

from langchain.llms import OpenAI

llm = OpenAI(
    model_name="qwen",
    temperature=0,
    openai_api_base="http://192.168.0.53:7891/v1",
    openai_api_key="xxx",
)

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

PROMPT_REACT = """你是一个可以调用外部工具的助手，可以使用的工具包括：
{tools_text}
如果使用工具请遵循以下格式回复：
```
Thought: 思考你当前步骤需要解决什么问题，是否需要使用工具
Action: 工具名称，你的工具必须从 [{tools_name_text}] 选择
ActionInput: 工具输入参数
```
工具返回按照以下格式回复：
```
Response: 调用工具后的结果
```
如果你已经知道了答案，或者你不需要工具，请遵循以下格式回复
```
Thought: 给出最终答案的思考过程
FinalAnswer: 最终答案
```
开始!

问题: {query}"""


def llm_with_plugin(prompt: str, history, list_of_plugin_info=()):
    """
    Args:
        prompt: 用户的最新一个问题。
        history: 用户与模型的对话历史，是一个 list，
            list 中的每个元素为 {"user": "用户输入", "bot": "模型输出"} 的一轮对话。
            最新的一轮对话放 list 末尾。不包含最新一个问题。
        list_of_plugin_info: 候选插件列表，是一个 list，list 中的每个元素为一个插件的关键信息。
            比如 list_of_plugin_info = [plugin_info_0, plugin_info_1, plugin_info_2]，
            其中 plugin_info_0, plugin_info_1, plugin_info_2 这几个样例见本文档前文。

    Returns: 模型对用户最新一个问题的回答。
    """
    chat_history = [(x["user"], x["bot"]) for x in history] + [(prompt, '')]

    # 需要让模型进行续写的初始文本
    planning_prompt = build_input_text(chat_history, list_of_plugin_info)

    text = ""
    while True:
        output = llm(planning_prompt + text, stop=["Response:"])
        action, action_input, output = parse_latest_plugin_call(output)
        if action:  # 需要调用插件
            # action、action_input 分别为需要调用的插件代号、输入参数
            # observation是插件返回的结果，为字符串
            observation = call_plugin(action, action_input)
            output += f"\nResponse: {observation}\nThought:"
            text += output
        else:  # 生成结束，并且不再需要调用插件
            text += output
            break

    new_history = []
    new_history.extend(history)
    new_history.append({"user": prompt, "bot": text})
    return text, new_history


def build_input_text(chat_history, list_of_plugin_info) -> str:
    """ 将对话历史、插件信息聚合成一段初始文本 """
    tools_text = []
    for plugin_info in list_of_plugin_info:
        tool = TOOL_DESC.format(
            name_for_model=plugin_info["name_for_model"],
            name_for_human=plugin_info["name_for_human"],
            description_for_model=plugin_info["description_for_model"],
            parameters=json.dumps(plugin_info["parameters"], ensure_ascii=False),
        )
        if plugin_info.get("args_format", "json") == "json":
            tool += " Format the arguments as a JSON object."
        elif plugin_info['args_format'] == 'code':
            tool += " Enclose the code within triple backticks (`) at the beginning and end of the code."
        else:
            raise NotImplementedError
        tools_text.append(tool)
    tools_text = '\n\n'.join(tools_text)

    # 候选插件的代号
    tools_name_text = ", ".join([plugin_info["name_for_model"] for plugin_info in list_of_plugin_info])

    prompt = ""
    for i, (query, response) in enumerate(chat_history):
        if list_of_plugin_info:  # 如果有候选插件
            # 倒数第一轮或倒数第二轮对话填入详细的插件信息，但具体什么位置填可以自行判断
            if (len(chat_history) == 1) or (i == len(chat_history) - 2):
                query = PROMPT_REACT.format(
                    tools_text=tools_text,
                    tools_name_text=tools_name_text,
                    query=query,
                )
        query = query.lstrip("\n").rstrip()
        response = response.lstrip("\n").rstrip()
        # 使用续写模式（text completion）时，需要用如下格式区分用户和AI：
        prompt += f"\n<s><|User|>:{query}<eoh>"
        prompt += f"\n<|Bot|>:{response}<eoa>"

    assert prompt.endswith("\n<|Bot|>:<eoa>")
    prompt = prompt[1: -len("<eoa>")]
    return prompt


def text_completion(input_text: str, stop_words) -> str:  # 作为一个文本续写模型来使用
    return llm(input_text, stop=stop_words)  # 续写 input_text 的结果，不包含 input_text 的内容


def parse_latest_plugin_call(text):
    plugin_name, plugin_args = "", ""
    i = text.rfind("\nAction:")
    j = text.rfind("\nActionInput:")
    k = text.rfind("\nResponse:")

    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + "\nResponse:"  # Add it back.
        k = text.rfind("\nResponse:")
        plugin_name = text[i + len("\nAction:"): j].strip()
        plugin_args = text[j + len("\nActionInput:"): k].strip()
        text = text[:k]
    return plugin_name, plugin_args, text


def call_plugin(plugin_name: str, plugin_args: str) -> str:
    """ 请开发者自行完善这部分内容。这里的参考实现仅是 demo 用途，非生产用途 """
    if plugin_name == "image_gen":
        import urllib.parse

        prompt = json.loads(plugin_args)["prompt"]
        prompt = urllib.parse.quote(prompt)
        return json.dumps({"image_url": f"https://image.pollinations.ai/prompt/{prompt}"}, ensure_ascii=False)
    elif plugin_name == "calculate_quad":
        from scipy import integrate

        def calculate_quad(formula_str: str, a: float, b: float) -> float:
            """ 计算数值积分 """
            return integrate.quad(eval('lambda x: ' + formula_str.split("=")[-1]), a, b)[0]

        plugin_args = json.loads(plugin_args)
        for k in ["a", "b"]:
            if k in plugin_args:
                plugin_args[k] = float(plugin_args[k])
        return calculate_quad(**plugin_args)
    else:
        raise NotImplementedError


def test():
    tools = [
        {
            "name_for_human": "文生图",
            "name_for_model": "image_gen",
            "description_for_model": "文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL",
            "parameters": [
                {
                    "name": "prompt",
                    "description": "英文关键词，描述了希望图像具有什么内容",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
        {
            "name_for_human": "积分计算器",
            "name_for_model": "calculate_quad",
            "description_for_model": "积分计算器是一个可以计算给定区间内函数定积分数值的工具。",
            "parameters": [
                {
                    "name": 'formula_str',
                    "description": '一个数学函数的表达式，例如x**2+x',
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "a",
                    "description": "积分区间的左端点，例如1.0",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "b",
                    "description": "积分区间的右端点，例如5.0",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
        },
    ]
    history = []
    for query in ["你好", "给我画个可爱的小猫吧，最好是黑猫", "函数f(x)=x**2在区间[0,5]上的定积分是多少？"]:
        print(f"User's Query:\n{query}\n")
        response, history = llm_with_plugin(prompt=query, history=history, list_of_plugin_info=tools)
        print(f"InternLM's Response:\n{response}\n")


if __name__ == "__main__":
    test()

"""
User's Query:
你好

InternLM's Response:
Thought: 这是一个问候语，不需要调用任何工具。
FinalAnswer: 您好！有什么我可以帮助您的吗？

User's Query:
给我画个可爱的小猫吧，最好是黑猫

InternLM's Response:
Thought: 这是一道基于语言的图像生成任务，需要使用文生图 API。
Action: image_gen
ActionInput: {"prompt": "一只黑猫"}
Response: {"image_url": "https://image.pollinations.ai/prompt/%E4%B8%80%E5%8F%AA%E9%BB%91%E7%8C%AB"}
Thought: Base on the result of the code, the answer is:
FinalAnswer: 根据您的要求，我使用文生图 API 为您生成了一只黑猫。以下是它的图片链接：https://image.pollinations.ai/prompt/%E4%B8%80%E5%8F%AA%E9%BB%91%E7%8C%AB


User's Query:
函数f(x)=x**2在区间[0,5]上的定积分是多少？

InternLM's Response:
Thought: 这是一道基于积分的数学问题，需要使用积分计算器来计算。
Action: calculate_quad
ActionInput: {"formula_str": "f(x)=x**2", "a": "0", "b": "5"}
Response: 41.66666666666666
Thought: Base on the result of the code, the answer is:
FinalAnswer: 函数f(x)=x**2在区间[0,5]上的定积分是41.66666666666666。
"""
