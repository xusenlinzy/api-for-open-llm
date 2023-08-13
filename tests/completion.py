import openai

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.0.53:7891/v1"

# List models API
models = openai.Model.list()
print("Models:", models)

model = models["data"][0]["id"]

# Chat completion API
completion = openai.Completion.create(
    model=model,
    prompt="感冒了怎么办"
)

print("Completion results:")
print(completion)

completion = openai.Completion.create(
    model=model,
    prompt="感冒了怎么办",
    stream=True
)

print("Completion streaming results:")
for c in completion:
    print(c.choices[0].text)
