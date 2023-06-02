## General Chatbot

```shell
python chat.py --model_name chatglm --api_base http://192.168.0.xx:80/v1
```

## Document Chatbot

```shell
python chat.py --model_name chatglm --api_base http://192.168.0.xx:80/v1
```

## Web Demo

```shell
python web_demo.py
```

支持通用问答、知识库问答和数据库问答三种模式

## Docker

```shell
docker run -it -p 7860:7860 --platform=linux/amd64 \
	registry.hf.space/xusenlin-openllm:latest python app.py
```
