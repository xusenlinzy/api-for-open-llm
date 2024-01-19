import json
from typing import (
    Optional,
    List,
    AsyncIterator,
)

from aiohttp import ClientSession
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError
from text_generation import AsyncClient
from text_generation.errors import parse_error
from text_generation.types import Request, Parameters
from text_generation.types import Response, StreamResponse

from api.adapter import get_prompt_adapter
from api.utils.compat import model_dump


class TGIEngine:
    def __init__(
        self,
        model: AsyncClient,
        model_name: str,
        prompt_name: Optional[str] = None,
    ):
        """
        Initializes the TGIEngine object.

        Args:
            model: The AsyncLLMEngine object.
            model_name: The name of the model.
            prompt_name: The name of the prompt (optional).
        """
        self.model = model
        self.model_name = model_name.lower()
        self.prompt_name = prompt_name.lower() if prompt_name is not None else None
        self.prompt_adapter = get_prompt_adapter(self.model_name, prompt_name=self.prompt_name)

    def apply_chat_template(
        self, messages: List[ChatCompletionMessageParam],
    ) -> str:
        """
        Applies a chat template to the given messages and returns the processed output.

        Args:
            messages: A list of ChatCompletionMessageParam objects representing the chat messages.

        Returns:
            str: The processed output as a string.
        """
        return self.prompt_adapter.apply_chat_template(messages)

    async def generate(
        self,
        prompt: str,
        do_sample: bool = True,
        max_new_tokens: int = 20,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        decoder_input_details: bool = True,
        top_n_tokens: Optional[int] = None,
    ) -> Response:
        """
        Given a prompt, generate the following text asynchronously

        Args:
            prompt (`str`):
                Input text
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            best_of (`int`):
                Generate best_of sequences and return the one if the highest token logprobs
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of the highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            decoder_input_details (`bool`):
                Return the decoder input token logprobs and ids
            top_n_tokens (`int`):
                Return the `n` most likely tokens at each step

        Returns:
            Response: generated response
        """
        # Validate parameters
        parameters = Parameters(
            best_of=best_of,
            details=True,
            decoder_input_details=decoder_input_details,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            top_n_tokens=top_n_tokens,
        )
        request = Request(inputs=prompt, stream=False, parameters=parameters)

        async with ClientSession(
            headers=self.model.headers, cookies=self.model.cookies, timeout=self.model.timeout
        ) as session:
            async with session.post(f"{self.model.base_url}/generate", json=model_dump(request)) as resp:
                payload = await resp.json()

                if resp.status != 200:
                    raise parse_error(resp.status, payload)
                return Response(**payload)

    async def generate_stream(
        self,
        prompt: str,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        best_of: Optional[int] = 1,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        top_n_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamResponse]:
        """
        Given a prompt, generate the following stream of tokens asynchronously

        Args:
            prompt (`str`):
                Input text
            do_sample (`bool`):
                Activate logits sampling
            max_new_tokens (`int`):
                Maximum number of generated tokens
            best_of (`int`):
                Generate best_of sequences and return the one if the highest token logprobs
            repetition_penalty (`float`):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            return_full_text (`bool`):
                Whether to prepend the prompt to the generated text
            seed (`int`):
                Random sampling seed
            stop_sequences (`List[str]`):
                Stop generating tokens if a member of `stop_sequences` is generated
            temperature (`float`):
                The value used to module the logits distribution.
            top_k (`int`):
                The number of the highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`):
                If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
                higher are kept for generation.
            truncate (`int`):
                Truncate inputs tokens to the given size
            typical_p (`float`):
                Typical Decoding mass
                See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
            watermark (`bool`):
                Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
            top_n_tokens (`int`):
                Return the `n` most likely tokens at each step

        Returns:
            AsyncIterator: stream of generated tokens
        """
        # Validate parameters
        parameters = Parameters(
            best_of=best_of,
            details=True,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            return_full_text=return_full_text,
            seed=seed,
            stop=stop_sequences if stop_sequences is not None else [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            truncate=truncate,
            typical_p=typical_p,
            watermark=watermark,
            top_n_tokens=top_n_tokens,
        )
        request = Request(inputs=prompt, parameters=parameters)

        async with ClientSession(
            headers=self.model.headers, cookies=self.model.cookies, timeout=self.model.timeout
        ) as session:
            async with session.post(f"{self.model.base_url}/generate_stream", json=model_dump(request)) as resp:
                if resp.status != 200:
                    raise parse_error(resp.status, await resp.json())

                # Parse ServerSentEvents
                async for byte_payload in resp.content:
                    # Skip line
                    if byte_payload == b"\n":
                        continue

                    payload = byte_payload.decode("utf-8")

                    # Event data
                    if payload.startswith("data:"):
                        # Decode payload
                        json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                        # Parse payload
                        try:
                            response = StreamResponse(**json_payload)
                        except ValidationError:
                            # If we failed to parse the payload, then it is an error payload
                            raise parse_error(resp.status, json_payload)
                        yield response

    @property
    def stop(self):
        """
        Gets the stop property of the prompt adapter.

        Returns:
            The stop property of the prompt adapter, or None if it does not exist.
        """
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None
