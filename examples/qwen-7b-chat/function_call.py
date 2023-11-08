# Reference: https://openai.com/blog/function-calling-and-other-api-updates

from openai import OpenAI
from loguru import logger

client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.20.59:7891/v1/",
)


def call_qwen(messages, functions=None):
    logger.info(messages)
    if functions:
        response = client.chat.completions.create(
            model="Qwen", messages=messages, functions=functions
        )
    else:
        response = client.chat.completions.create(model="Qwen", messages=messages)
    logger.info(response.dict())
    logger.info(response.choices[0].message.content)
    return response


def run_test_1():
    messages = [{"role": "user", "content": "你好"}]
    call_qwen(messages)
    messages.append({"role": "assistant", "content": "你好！很高兴为你提供帮助。"})

    messages.append({"role": "user", "content": "给我讲一个年轻人奋斗创业最终取得成功的故事。故事只能有一句话。"})
    call_qwen(messages)
    messages.append(
        {
            "role": "assistant",
            "content": "故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。李明想要成为一名成功的企业家。……",
        }
    )

    messages.append({"role": "user", "content": "给这个故事起一个标题"})
    call_qwen(messages)


def run_test_2():
    functions = [
        {
            "name_for_human": "谷歌搜索",
            "name_for_model": "google_search",
            "description_for_model": "谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。"
            + " Format the arguments as a JSON object.",
            "parameters": [
                {
                    "name": "search_query",
                    "description": "搜索关键词或短语",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
        {
            "name_for_human": "文生图",
            "name_for_model": "image_gen",
            "description_for_model": "文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL。"
            + " Format the arguments as a JSON object.",
            "parameters": [
                {
                    "name": "prompt",
                    "description": "英文关键词，描述了希望图像具有什么内容",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
    ]

    messages = [{"role": "user", "content": "你好"}]
    call_qwen(messages, functions)
    messages.append(
        {"role": "assistant", "content": "你好！很高兴见到你。有什么我可以帮忙的吗？"},
    )

    messages.append({"role": "user", "content": "谁是周杰伦"})
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "Thought: 我应该使用Google搜索查找相关信息。",
            "function_call": {
                "name": "google_search",
                "arguments": '{"search_query": "周杰伦"}',
            },
        }
    )

    messages.append(
        {
            "role": "function",
            "name": "google_search",
            "content": "Jay Chou is a Taiwanese singer.",
        }
    )
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "周杰伦（Jay Chou）是一位来自台湾的歌手。",
        },
    )

    messages.append({"role": "user", "content": "他老婆是谁"})
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "Thought: 我应该使用Google搜索查找相关信息。",
            "function_call": {
                "name": "google_search",
                "arguments": '{"search_query": "周杰伦 老婆"}',
            },
        }
    )

    messages.append(
        {"role": "function", "name": "google_search", "content": "Hannah Quinlivan"}
    )
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "周杰伦的老婆是Hannah Quinlivan。",
        },
    )

    messages.append({"role": "user", "content": "给我画个可爱的小猫吧，最好是黑猫"})
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "Thought: 我应该使用文生图API来生成一张可爱的小猫图片。",
            "function_call": {
                "name": "image_gen",
                "arguments": '{"prompt": "cute black cat"}',
            },
        }
    )

    messages.append(
        {
            "role": "function",
            "name": "image_gen",
            "content": '{"image_url": "https://image.pollinations.ai/prompt/cute%20black%20cat"}',
        }
    )
    call_qwen(messages, functions)


def run_test_3():
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]

    messages = [
        {
            "role": "user",
            # Note: The current version of Qwen-7B-Chat (as of 2023.08) performs okay with Chinese tool-use prompts,
            # but performs terribly when it comes to English tool-use prompts, due to a mistake in data collecting.
            "content": "波士顿天气如何？",
        }
    ]
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": "get_current_weather",
                "arguments": '{"location": "Boston, MA"}',
            },
        },
    )

    messages.append(
        {
            "role": "function",
            "name": "get_current_weather",
            "content": '{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
        }
    )
    call_qwen(messages, functions)


if __name__ == "__main__":
    logger.info("### Test Case 1 - No Function Calling (普通问答、无函数调用) ###")
    run_test_1()
    logger.info("### Test Case 2 - Use Qwen-Style Functions (函数调用，千问格式) ###")
    run_test_2()
    logger.info("### Test Case 3 - Use GPT-Style Functions (函数调用，GPT格式) ###")
    run_test_3()
