import json

import openai

openai.api_base = "http://192.168.20.59:7891/v1"
openai.api_key = "xxx"


tools = [
    {
        "name": "track",
        "description": "追踪指定股票的实时价格",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "description": "需要追踪的股票代码"
                }
            },
            "required": ['symbol']
        }
    },
    {
        "name": "text-to-speech",
        "description": "将文本转换为语音",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "description": "需要转换成语音的文本"
                },
                "voice": {
                    "description": "要使用的语音类型（男声、女声等）"
                },
                "speed": {
                    "description": "语音的速度（快、中等、慢等）"
                }
            },
            "required": ['text']
        }
    }
]

system_info = {
    "role": "system",
    "content": "Answer the following questions as best as you can. You have access to the following tools:",
    "tools": tools,
}


def test():
    messages = [
        system_info,
        {
            "role": "user",
            "content": "帮我查询股票10111的价格",
        }
    ]
    response = openai.ChatCompletion.create(
        model="chatglm3",
        messages=messages,
        temperature=0,
    )
    print(response.choices[0].message.content)

    messages = response.choices[0].history  # 获取历史对话信息
    messages.append(
        {
            "role": "observation",
            "content": json.dumps({"price": 12412}, ensure_ascii=False),  # 调用函数返回结果
        }
    )

    response = openai.ChatCompletion.create(
        model="chatglm3",
        messages=messages,
        temperature=0,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    test()
