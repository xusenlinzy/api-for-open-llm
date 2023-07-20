import gc
import re
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from api.constants import ErrorCode
from api.prompt_adapter import get_prompt_adapter

server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


invalid_score_processor = InvalidScoreLogitsProcessor()


def process_response(response):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    return response


@torch.inference_mode()
def generate_stream_chatglm(
    model,
    tokenizer,
    params,
    device,
    context_len=2048,
    stream_interval=2,
):
    prompt = params["prompt"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = params.get("echo", True)

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_echo_len = len(inputs["input_ids"][0])

    gen_kwargs = {
        "max_length": max_new_tokens + input_echo_len,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [invalid_score_processor],
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len = 0
    for total_ids in model.stream_generate(**inputs, **gen_kwargs):
        total_ids = total_ids.tolist()[0]
        total_len = len(total_ids)
        if echo:
            output_ids = total_ids
        else:
            output_ids = total_ids[input_echo_len:]
        response = tokenizer.decode(output_ids)
        response = process_response(response)

        yield {
            "text": response,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
            "finish_reason": None,
        }

    # TODO: ChatGLM stop when it reach max length
    # Only last stream result contains finish_reason, we set finish_reason as stop
    ret = {
        "text": response,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
        "finish_reason": "stop",
    }
    yield ret

    gc.collect()
    torch.cuda.empty_cache()


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op, so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def is_partial_stop(output: str, stop_str: str):
    """Check whether the output contains a partial stop str."""
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    params,
    device: str,
    context_len: int,
    stream_interval: int = 2,
):
    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompt).input_ids

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    past_key_values = None
    sent_interrupt = False
    first_tokens = None
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:  # decoding
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            if i == 0:
                first_token_probs = torch.softmax(last_token_logits, dim=-1)
                first_token_probs, first_token_indices = torch.topk(first_token_probs, k=10, largest=True, sorted=True)
                topk_tokens = [tokenizer.decode(int(i)) for i in first_token_indices]

                first_tokens = {}
                for t, p in zip(topk_tokens, first_token_probs.tolist()):
                    if t in first_tokens and p < first_tokens[t]:
                        continue
                    first_tokens[t] = p

            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                        "first_tokens": first_tokens
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
            "first_tokens": first_tokens
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


SEQUENCE_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_position_embeddings",
    "max_seq_len",
    "model_max_length",
]


def get_context_length(config):
    """Get the context length of a model from a huggingface model config."""
    for key in SEQUENCE_LENGTH_KEYS:
        if hasattr(config, key):
            val = getattr(config, key)
            if val is not None:
                return val
    return 2048


class ModelServer:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        model_name,
        context_len: Optional[int] = None,
        stream_interval: Optional[int] = 2,
        prompt_name: Optional[str] = None,
    ):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name.lower()
        self.prompt_name = prompt_name.lower() if prompt_name is not None else None
        self.stream_interval = stream_interval

        if context_len is None:
            self.context_len = get_context_length(self.model.config)
        else:
            self.context_len = context_len

        # generate_stream
        self.has_chat_fct = "chatglm" in self.model_name
        if "chatglm" in self.model_name:
            self.generate_stream_func = generate_stream_chatglm
        else:
            self.generate_stream_func = generate_stream

        self.prompt_adapter = get_prompt_adapter(self.model_name, prompt_name=self.prompt_name)

    def count_token(self, params):
        prompt = params["prompt"]
        input_ids = self.tokenizer(prompt).input_ids
        input_echo_len = len(input_ids)

        ret = {
            "count": input_echo_len,
            "error_code": 0,
        }
        return ret

    def generate_prompt(self, messages):
        return self.prompt_adapter.generate_prompt(messages)

    def generate_stream_gate(self, params):
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
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield ret

    def generate_gate(self, params):
        if isinstance(params["prompt"], list):
            params["prompt"] = self.generate_prompt(params["prompt"])

        try:
            ret = {"text": "", "error_code": 0}
            for output in self.generate_stream_func(
                self.model,
                self.tokenizer,
                params,
                self.device,
                self.context_len,
                self.stream_interval,
            ):
                ret["text"] = output["text"]

            if "usage" in output:
                ret["usage"] = output["usage"]
            if "finish_reason" in output:
                ret["finish_reason"] = output["finish_reason"]
            if "logprobs" in output:
                ret["logprobs"] = output["logprobs"]

        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }

        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret

    @torch.inference_mode()
    def get_embeddings(self, params):
        try:
            tokenizer = self.tokenizer
            is_llama = "llama" in str(type(self.model))  # vicuna support batch inference
            is_chatglm = "chatglm" in self.model_name
            is_t5 = "t5" in str(type(self.model))
            if is_llama:
                encoding = tokenizer.batch_encode_plus(
                    params["input"], padding=True, return_tensors="pt"
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                model_output = self.model(
                    input_ids, attention_mask, output_hidden_states=True
                )
                data = model_output.hidden_states[-1]
                mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
                masked_embeddings = data * mask
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                seq_length = torch.sum(mask, dim=1)
                embedding = sum_embeddings / seq_length
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)
                ret = {
                    "embedding": normalized_embeddings.tolist(),
                    "token_num": torch.sum(attention_mask).item(),
                }
            else:
                embedding = []
                token_num = 0
                for text in params["input"]:
                    input_ids = tokenizer.encode(text, return_tensors="pt").to(
                        self.device
                    )
                    if is_t5:
                        model_output = self.model(input_ids, decoder_input_ids=input_ids)
                    else:
                        model_output = self.model(input_ids, output_hidden_states=True)
                    if is_chatglm:
                        data = (model_output.hidden_states[-1].transpose(0, 1))[0]
                    elif is_t5:
                        data = model_output.encoder_last_hidden_state[0]
                    else:
                        data = model_output.hidden_states[-1][0]
                    data = F.normalize(torch.mean(data, dim=0), p=2, dim=0)
                    embedding.append(data.tolist())
                    token_num += len(input_ids[0])
                ret = {
                    "embedding": embedding,
                    "token_num": token_num,
                }
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret

    @torch.inference_mode()
    def get_other_embeddings(self, client, params):
        try:
            embeddings = client.encode(params["input"], normalize_embeddings=True)
            ret = {
                "embedding": embeddings.tolist(),
                "token_num": sum([len(i) for i in params["input"]]),
            }
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret

    @property
    def stop(self):
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None
