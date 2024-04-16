# Langchain LLM

## Get Started

### Install

```shell
pip install langchain_llm==0.4.15
```

## Inference Usage

### HuggingFace Inference

**Completion Usage**

```python
from langchain_llm import HuggingFaceLLM

llm = HuggingFaceLLM(
    model_name="qwen-7b-chat",
    model_path="/data/checkpoints/Qwen-7B-Chat",
    load_model_kwargs={"device_map": "auto"},
)

# invoke method
prompt = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n"
print(llm.invoke(prompt, stop=["<|im_end|>"]))

# Token Streaming
for chunk in llm.stream(prompt, stop=["<|im_end|>"]):
    print(chunk, end="", flush=True)

# openai usage
print(llm.call_as_openai(prompt, stop=["<|im_end|>"]))

# Streaming
for chunk in llm.call_as_openai(prompt, stop=["<|im_end|>"], stream=True):
    print(chunk.choices[0].text, end="", flush=True)
```

**Chat Completion Usage**

```python
from langchain_llm import ChatHuggingFace

chat_llm = ChatHuggingFace(llm=llm)

# invoke method
query = "你是谁？"
print(chat_llm.invoke(query))

# Token Streaming
for chunk in chat_llm.stream(query):
    print(chunk.content, end="", flush=True)

# openai usage
messages = [
    {"role": "user", "content": query}
]
print(chat_llm.call_as_openai(messages))

# Streaming
for chunk in chat_llm.call_as_openai(messages, stream=True):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### VLLM Inference

**Completion Usage**

```python
from langchain_llm import VLLM

llm = VLLM(
    model_name="qwen", 
    model="/data/checkpoints/Qwen-7B-Chat", 
    trust_remote_code=True,
)

# invoke method
prompt = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n"
print(llm.invoke(prompt, stop=["<|im_end|>"]))

# openai usage
print(llm.call_as_openai(prompt, stop=["<|im_end|>"]))
```

**Chat Completion Usage**

```python
from langchain_llm import ChatVLLM

chat_llm = ChatVLLM(llm=llm)

# invoke method
query = "你是谁？"
print(chat_llm.invoke(query))

# openai usage
messages = [
    {"role": "user", "content": query}
]
print(chat_llm.call_as_openai(messages))
```


## Custom Chat template

```python
from langchain_llm import BaseTemplate, ChatHuggingFace

class CustomTemplate(BaseTemplate):
    
    @property
    def template(self) -> str:
        return (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

chat_llm = ChatHuggingFace(
    llm=llm, 
    prompt_adapter=CustomTemplate()
)
```

## Load Model Kwargs

+ `model_name_or_path`: model name or path.


+ `use_fast_tokenizer`: default false.


+ `device_map`: "auto"、"cuda:0" etc.


+ `dtype`: "half", "bfloat16", "float32".


+ `load_in_8bit`: Load model in 8 bit.


+ `load_in_4bit`: Load model in 4 bit.


+ `rope_scaling`: Which scaling strategy should be adopted for the RoPE embeddings. Literal["linear", "dynamic"].


+ `flash_attn`: Enable FlashAttention-2.

## Merge Lora model

```python
from langchain_llm import apply_lora

apply_lora("base_model_path", "lora_path", "target_model_path")
```
