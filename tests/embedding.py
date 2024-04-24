from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.20.159:8000/v1/",
)


# compute the embedding of the text
embedding = client.embeddings.create(
    input="你好",
    model="aspire/acge_text_embedding",
    dimensions=384,
)
print(len(embedding.data[0].embedding))
