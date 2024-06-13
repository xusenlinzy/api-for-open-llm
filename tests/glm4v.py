from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.0.59:7891/v1",
)

stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "这张图片是什么地方？"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # Either an url or base64
                        "url": "http://djclub.cdn.bcebos.com/uploads/images/pageimg/20230325/64-2303252115313.jpg"
                    }
                }
            ]
        }
    ],
    model="glm-4v-9b",
    stream=True,
)
for part in stream:
    print(part.choices[0].delta.content or "", end="", flush=True)
