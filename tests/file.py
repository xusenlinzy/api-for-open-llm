import requests
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.0.59:7891/v1/",
)

print(client.files.list())


uf = client.files.create(
    file=open("../README.md", "rb"),
    purpose="chat",
)
print(uf)


print(
    requests.post(
        url="http://192.168.0.59:7891/v1/files/split",
        json={"file_id": uf.id},
    ).json()
)

df = client.files.delete(file_id=uf.id)
print(df)
