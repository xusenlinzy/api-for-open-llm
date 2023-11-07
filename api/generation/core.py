import traceback
from typing import Optional, List, Union

import torch
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from api.apapter import get_prompt_adapter
from api.generation.baichuan import check_is_baichuan
from api.generation.chatglm import generate_stream_chatglm, check_is_chatglm, generate_stream_chatglm_v3
from api.generation.qwen import check_is_qwen
from api.generation.stream import generate_stream, generate_stream_v2
from api.generation.utils import get_context_length
from api.generation.xverse import check_is_xverse
from api.utils.constants import ErrorCode
from transformers import PreTrainedModel, PreTrainedTokenizer

server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)


class ModelServer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Union[str, torch.device],
        model_name: str,
        context_len: Optional[int] = None,
        stream_interval: Optional[int] = 2,
        prompt_name: Optional[str] = None,
        use_streamer_v2: Optional[bool] = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device if hasattr(model, "device") else device

        self.model_name = model_name.lower()
        self.prompt_name = prompt_name.lower() if prompt_name is not None else None
        self.stream_interval = stream_interval
        self.context_len = context_len

        self.construct_prompt = True
        self.generate_stream_func = generate_stream
        if "chatglm3" in self.model_name:
            logger.info("Using ChatGLM3 Model for Chat!")
            self.generate_stream_func = generate_stream_chatglm_v3
            self.construct_prompt = False if self.prompt_name is None else True
        else:
            if check_is_chatglm(self.model):
                logger.info("Using ChatGLM Model for Chat!")
                self.generate_stream_func = generate_stream_chatglm
            elif check_is_baichuan(self.model):
                logger.info("Using Baichuan Model for Chat!")
                self.construct_prompt = False if self.prompt_name is None else True
            elif check_is_qwen(self.model):
                logger.info("Using Qwen Model for Chat!")
                self.construct_prompt = False if self.prompt_name is None else True
                self.context_len = 8192 if self.context_len is None else self.context_len
            elif check_is_xverse(self.model):
                logger.info("Using Xverse Model for Chat!")
                self.construct_prompt = False if self.prompt_name is None else True

        self.prompt_adapter = get_prompt_adapter(self.model_name, prompt_name=self.prompt_name)
        self.use_streamer_v2 = use_streamer_v2

        if self.context_len is None:
            self.context_len = get_context_length(self.model.config)

        self.fix_tokenizer()

    def fix_tokenizer(self):
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token = "<|endoftext|>"
            logger.info("Add eos token: {}".format(self.tokenizer.eos_token))

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Add pad token: {}".format(self.tokenizer.pad_token))

    def generate_prompt(
        self, messages: List[ChatCompletionMessageParam]
    ) -> Union[str, List[ChatCompletionMessageParam]]:
        return self.prompt_adapter.apply_chat_template(messages) if self.construct_prompt else messages

    def generate_stream_gate(self, params):
        if self.use_streamer_v2:
            yield from self.generate_stream_gate_v2(params)
        else:
            yield from self.generate_stream_gate_v1(params)

    def generate_stream_gate_v1(self, params):
        if isinstance(params["prompt"], list):
            params["prompt"] = self.generate_prompt(params["prompt"])

        try:
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                self.stream_interval,
            ):
                ret = {
                    "text": output["text"],
                    "error_code": 0,
                }
                if "usage" in output:
                    ret["usage"] = output["usage"]
                if "finish_reason" in output:
                    ret["finish_reason"] = output["finish_reason"]
                if "logprobs" in output:
                    ret["logprobs"] = output["logprobs"]
                yield ret

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield ret

        except (ValueError, RuntimeError) as e:
            traceback.print_exc(e)
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield ret

    def generate_stream_gate_v2(self, params):
        if isinstance(params["prompt"], list):
            params["prompt"] = self.generate_prompt(params["prompt"])

        try:
            yield from generate_stream_v2(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
            )

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield ret

        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield ret

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return x

    @property
    def stop(self):
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None
