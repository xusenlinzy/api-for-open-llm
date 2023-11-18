import json

import requests

base_url = "http://192.168.20.59:9009"


def create_chat_completion(model, messages, stream=False):
    data = {
        "model": model,  # 模型名称
        "messages": messages,  # 会话历史
        "stream": stream,  # 是否流式响应
        "max_tokens": 100,  # 最多生成字数
        "temperature": 0.8,  # 温度
        "top_p": 0.8,  # 采样概率
    }

    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=stream)
    if response.status_code == 200:
        if stream:
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')[6:]
                    try:
                        response_json = json.loads(decoded_line)
                        content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        print(content)
                    except:
                        print("Special Token:", decoded_line)
        else:
            # 处理非流式响应
            decoded_line = response.json()
            print(decoded_line)
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
            print(content)
    else:
        print("Error:", response.status_code)
        return None


if __name__ == "__main__":
    chat_messages = [
        {
            "role": "user",
            "content": "你好，给我讲一个故事，大概100字"
        }
    ]
    create_chat_completion("baichuan2-13b", chat_messages, stream=False)
