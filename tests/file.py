from openai import OpenAI
import requests

client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.20.159:8000/v1",
)

print(client.files.list())


uf = client.files.create(
    file=open("../README.md", "rb"),
    purpose="chat",
)
print(uf)


df = client.files.delete(file_id=uf.id)
print(df)

print(
    requests.post(
        url="http://192.168.20.159:8000/v1/files/split",
        json={},
        files={"file": open("../README.md", "rb")}
    ).json()
)
