from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.20.44:7861/v1/",
)


# Chat completion API
completion = client.completions.create(
    model="gpt-3.5-turbo",
    prompt="感冒了怎么办",
)
print(completion)


stream = client.completions.create(
    model="gpt-3.5-turbo",
    prompt="感冒了怎么办",
    stream=True,
)
for part in stream:
    print(part.choices[0].text or "", end="", flush=True)
