from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.20.44:7861/v1/",
)


# List models API
models = client.models.list()
print(models.model_dump())


# Chat completion API
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "感冒了怎么办",
        }
    ],
    model="gpt-3.5-turbo",
)
print(chat_completion)


stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "感冒了怎么办",
        }
    ],
    model="gpt-3.5-turbo",
    stream=True,
)
for part in stream:
    print(part.choices[0].delta.content or "", end="", flush=True)
