import asyncio
from typing import (
    Optional,
    Dict,
)
from typing import Sequence as GenericSequence

from loguru import logger
from openai.types.completion_choice import Logprobs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import get_tokenizer

from api.templates import get_template


class VllmEngine:
    def __init__(
        self,
        model: AsyncLLMEngine,
        model_name: str,
        template_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.model_name = model_name.lower()
        self.template_name = template_name.lower() if template_name else self.model_name

        logger.info(f"Using {self.model_name} Model for Chat!")

        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        if event_loop is not None and event_loop.is_running():
            # If the current is instanced by Ray Serve,
            # there is already a running event loop
            event_loop.create_task(self._post_init())
        else:
            # When using single vLLM without engine_use_ray
            asyncio.run(self._post_init())

    async def _post_init(self):
        engine_model_config = await self.model.get_model_config()
        self.max_model_len = engine_model_config.max_model_len

        # A separate tokenizer to map token IDs to strings.
        self.tokenizer = get_tokenizer(
            engine_model_config.tokenizer,
            tokenizer_mode=engine_model_config.tokenizer_mode,
            trust_remote_code=engine_model_config.trust_remote_code,
        )

        self.template = get_template(self.template_name, self.tokenizer, self.max_model_len)
        logger.info(f"Using {self.template} for chat!")

    def create_completion_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[Optional[Dict[int, Logprob]]],
        num_output_top_logprobs: int,
        initial_text_offset: int = 0,
    ) -> Logprobs:
        """Create OpenAI-style logprobs."""
        logprobs = Logprobs(
            text_offset=[], token_logprobs=[], tokens=[], top_logprobs=[]
        )

        last_token_len = 0
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = self.tokenizer.decode(token_id)
                logprobs.tokens.append(token)
                logprobs.token_logprobs.append(None)
                logprobs.top_logprobs.append(None)
            else:
                token = self._get_decoded_token(step_top_logprobs[token_id], token_id)
                token_logprob = max(step_top_logprobs[token_id].logprob, -9999.0)
                logprobs.tokens.append(token)
                logprobs.token_logprobs.append(token_logprob)

                # makes sure to add the top num_output_top_logprobs + 1
                # logprobs, as defined in the openai API
                # (cf. https://github.com/openai/openai-openapi/blob/
                # 893ba52242dbd5387a97b96444ee1c742cfce9bd/openapi.yaml#L7153)
                logprobs.top_logprobs.append({
                    # Convert float("-inf") to the
                    # JSON-serializable float that OpenAI uses
                    self._get_decoded_token(top_lp[1], top_lp[0]):
                        max(top_lp[1].logprob, -9999.0)
                    for i, top_lp in enumerate(step_top_logprobs.items())
                    if num_output_top_logprobs >= i
                })

            if len(logprobs.text_offset) == 0:
                logprobs.text_offset.append(initial_text_offset)
            else:
                logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
            last_token_len = len(token)

        return logprobs

    def _get_decoded_token(self, logprob: Logprob, token_id: int) -> str:
        if logprob.decoded_token is not None:
            return logprob.decoded_token
        return self.tokenizer.decode(token_id)
