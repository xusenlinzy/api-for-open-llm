from typing import (
    Optional,
    List,
    Union,
    Dict,
    Iterator,
    Any,
)

from llama_cpp import Llama
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
)
from openai.types.completion_usage import CompletionUsage

from api.adapter import get_prompt_adapter


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
        self,
        messages: List[ChatCompletionMessageParam],
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if self.prompt_adapter.function_call_available:
            messages = self.prompt_adapter.postprocess_messages(messages, functions, tools)
        return self.prompt_adapter.apply_chat_template(messages)

    def create_completion(self, prompt, **kwargs) -> Union[Iterator, Dict[str, Any]]:
        return self.model.create_completion(prompt, **kwargs)

    def _create_chat_completion(self, prompt, **kwargs) -> ChatCompletion:
        completion = self.create_completion(prompt, **kwargs)
        message = ChatCompletionMessage(
            role="assistant",
            content=completion["choices"][0]["text"].strip(),
        )
        choice = Choice(
            index=0,
            message=message,
            finish_reason="stop",
        )
        usage = CompletionUsage.model_validate(completion["usage"])
        return ChatCompletion(
            id="chat" + completion["id"],
            choices=[choice],
            created=completion["created"],
            model=completion["model"],
            object="chat.completion",
            usage=usage,
        )

    def _create_chat_completion_stream(self, prompt, **kwargs) -> Iterator:
        completion = self.create_completion(prompt, **kwargs)
        for i, output in enumerate(completion):
            _id, _created, _model = output["id"], output["created"], output["model"]
            if i == 0:
                choice = ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=""),
                    finish_reason=None,
                )
                yield ChatCompletionChunk(
                    id=_id,
                    choices=[choice],
                    created=_created,
                    model=_model,
                    object="chat.completion.chunk",
                )

            if output["choices"][0]["finish_reason"] is None:
                delta = ChoiceDelta(content=output["choices"][0]["text"])
            else:
                delta = ChoiceDelta()

            choice = ChunkChoice(
                index=0,
                delta=delta,
                finish_reason=output["choices"][0]["finish_reason"],
            )
            yield ChatCompletionChunk(
                id=_id,
                choices=[choice],
                created=_created,
                model=_model,
                object="chat.completion.chunk",
            )

    def create_chat_completion(self, prompt, **kwargs) -> Union[Iterator, ChatCompletion]:
        if kwargs.get("stream", False):
            chat_completion_or_chunks = self._create_chat_completion_stream(prompt, **kwargs)
        else:
            chat_completion_or_chunks = self._create_chat_completion(prompt, **kwargs)
        return chat_completion_or_chunks

    @property
    def stop(self):
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None
