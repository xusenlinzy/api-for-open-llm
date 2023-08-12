import openai

openai.api_base = "http://192.168.0.53:7891/v1"

# Enter any non-empty API key to pass the client library's check.
openai.api_key = "xxx"

# compute the embedding of the text
embedding = openai.Embedding.create(
    input="什么是chatgpt？",
    model="m3e-base"
)

print(embedding['data'][0]['embedding'])
