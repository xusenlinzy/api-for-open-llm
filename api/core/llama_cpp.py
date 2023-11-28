from typing import (
    Optional,
    List,
    Union,
    Dict,
    Iterator,
    Any,
)

from openai.types.chat import ChatCompletionMessageParam

from api.apapter import get_prompt_adapter
from llama_cpp import Llama


def _convert_text_completion_to_chat(completion, completion_index: int = 0):
    return {
        "id": "chat" + completion["id"],
        "object": "chat.completion",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [
            {
                "index": completion_index,
                "message": {
                    "role": "assistant",
                    "content": completion["choices"][0]["text"],
                },
                "finish_reason": completion["choices"][0]["finish_reason"],
            }
        ],
        "usage": completion["usage"],
    }


def _convert_text_completion_chunks_to_chat(chunks, completion_index: int = 0):
    for i, chunk in enumerate(chunks):
        if i == 0:
            yield {
                "id": "chat" + chunk["id"],
                "model": chunk["model"],
                "created": chunk["created"],
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": completion_index,
                        "delta": {
                            "role": "assistant",
                        },
                        "finish_reason": None,
                    }
                ],
            }
        yield {
            "id": "chat" + chunk["id"],
            "model": chunk["model"],
            "created": chunk["created"],
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": completion_index,
                    "delta": {
                        "content": chunk["choices"][0]["text"],
                    }
                    if chunk["choices"][0]["finish_reason"] is None else {},
                    "finish_reason": chunk["choices"][0]["finish_reason"],
                }
            ],
        }


def convert_completion_to_chat(completion_or_chunks, stream: bool = False, completion_index: int = 0):
    if stream:
        chunks = completion_or_chunks
        return _convert_text_completion_chunks_to_chat(chunks, completion_index)
    else:
        completion = completion_or_chunks
        return _convert_text_completion_to_chat(completion, completion_index)


class LlamaCppEngine:
    def __init__(
        self,
        model: Llama,
        model_name: str,
        prompt_name: Optional[str] = None,
    ):
        self.model = model
        self.model_name = model_name.lower()
        self.prompt_name = prompt_name.lower() if prompt_name is not None else None
        self.prompt_adapter = get_prompt_adapter(self.model_name, prompt_name=self.prompt_name)

    def apply_chat_template(
        self, messages: List[ChatCompletionMessageParam], **kwargs,
    ) -> str:
        return self.prompt_adapter.apply_chat_template(messages)

    def create_completion(self, prompt, **kwargs) -> Union[Iterator, Dict[str, Any]]:
        return self.model.create_completion(prompt, **kwargs)

    def create_chat_completion(self, prompt, **kwargs) -> Union[Iterator, Dict[str, Any]]:
        completion_or_chunks = self.create_completion(prompt, **kwargs)
        return convert_completion_to_chat(completion_or_chunks, stream=kwargs.get("stream", False))

    @property
    def stop(self):
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None
