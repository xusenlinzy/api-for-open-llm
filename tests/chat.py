import openai

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.0.53:7891/v1"

# List models API
models = openai.Model.list()
print("Models:", models)

model = models["data"][0]["id"]

# Chat completion API
chat_completion = openai.ChatCompletion.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "感冒了怎么办"
        },
    ]
)

print("Chat completion results:")
print(chat_completion)

chat_completion = openai.ChatCompletion.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "感冒了怎么办"
        },
    ],
    stream=True
)

print("Chat completion streaming results:")
for c in chat_completion:
    print(c.choices[0].delta.get("content", ""))
