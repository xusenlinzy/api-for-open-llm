import cohere

client = cohere.Client(api_key="none", base_url="http://192.168.20.44:7861/v1")

query = "人工智能"
corpus = [
    "人工智能",
    "AI",
    "我喜欢看电影",
    "如何学习自然语言处理？",
    "what's Artificial Intelligence?",
]

results = client.rerank(model="bce", query=query, documents=corpus, return_documents=True)
print(results.json(indent=4, ensure_ascii=False))
