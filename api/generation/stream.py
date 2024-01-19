import gc
import time
import uuid
from threading import Thread
from types import MethodType
from typing import (
    Iterable,
    Dict,
    Any,
)

import torch
from transformers import (
    TextIteratorStreamer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from api.generation.qwen import check_is_qwen
from api.generation.utils import (
    prepare_logits_processor,
    is_partial_stop,
    apply_stopping_strings,
)


@torch.inference_mode()
def generate_stream(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    params: Dict[str, Any],
):
    # Read parameters
    input_ids = params.get("inputs")
    prompt = params.get("prompt")
    model_name = params.get("model", "llm")
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_tokens", 256))
    logprobs = params.get("logprobs")
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop")

    stop_token_ids = params.get("stop_token_ids") or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    device = model.device
    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )
    else:
        start_ids = torch.as_tensor([input_ids], device=device)

    past_key_values, sent_interrupt = None, False
    token_logprobs = [None]  # The first token has no logprobs.
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"
    created: int = int(time.time())
    previous_text = ""
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

            if logprobs is not None:
                # Prefull logprobs for the prompt.
                shift_input_ids = start_ids[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = torch.log_softmax(shift_logits, dim=-1).tolist()
                for label_id, logit in zip(
                    shift_input_ids[0].tolist(), shift_logits[0]
                ):
                    token_logprobs.append(logit[label_id])

        else:  # decoding
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [output_ids if sent_interrupt else [token]], device=device
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=None if sent_interrupt else past_key_values,
                )
                sent_interrupt = False

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [output_ids if sent_interrupt else [token]], device=device
                    ),
                    use_cache=True,
                    past_key_values=None if sent_interrupt else past_key_values,
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
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]

        token = tokens[0]
        output_ids.append(token)

        if logprobs is not None:
            # Cannot use last_token_logits because logprobs is based on raw logits.
            token_logprobs.append(
                torch.log_softmax(logits[0, -1, :], dim=-1)[token].tolist()
            )

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % 2 == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len(prompt)
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=False if check_is_qwen(model) else True,  # fix for qwen react
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            ret_logprobs = None
            if logprobs is not None:
                ret_logprobs = {
                    "text_offset": [],
                    "tokens": [
                        tokenizer.decode(token)
                        for token in (
                            output_ids if echo else output_ids[input_echo_len:]
                        )
                    ],
                    "token_logprobs": token_logprobs if echo else token_logprobs[input_echo_len:],
                    "top_logprobs": [{}] * len(token_logprobs if echo else token_logprobs[input_echo_len:]),
                }
                # Compute text_offset
                curr_pos = 0
                for text in ret_logprobs["tokens"]:
                    ret_logprobs["text_offset"].append(curr_pos)
                    curr_pos += len(text)

            partially_stopped, finish_reason = False, None
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
                            if each_stop == "Observation:":
                                finish_reason = "function_call"
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if (not partially_stopped) and output and output[-1] != "�":
                delta_text = output[len(previous_text):]
                previous_text = output

                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "delta": delta_text,
                    "text": output,
                    "logprobs": ret_logprobs,
                    "finish_reason": finish_reason,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                }

        if stopped:
            break

    yield {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "delta": "",
        "text": output,
        "logprobs": ret_logprobs,
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_stream_v2(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    params: Dict[str, Any],
):
    input_ids = params.get("inputs")
    functions = params.get("functions")
    model_name = params.get("model", "llm")
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", 40))
    max_new_tokens = int(params.get("max_tokens", 256))

    stop_token_ids = params.get("stop_token_ids") or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)
    stop_strings = params.get("stop", [])

    input_echo_len = len(input_ids)
    device = model.device
    generation_kwargs = dict(
        input_ids=torch.tensor([input_ids], device=device),
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature <= 1e-5:
        generation_kwargs["do_sample"] = False
        generation_kwargs.pop("top_k")

    streamer = TextIteratorStreamer(
        tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs["streamer"] = streamer

    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text, func_call_found = "", False
    completion_id: str = f"cmpl-{str(uuid.uuid4())}"
    created: int = int(time.time())
    previous_text = ""
    for i, new_text in enumerate(streamer):
        generated_text += new_text
        if functions:
            _, func_call_found = apply_stopping_strings(generated_text, ["Observation:"])
        generated_text, stop_found = apply_stopping_strings(generated_text, stop_strings)

        if generated_text and generated_text[-1] != "�":
            delta_text = generated_text[len(previous_text):]
            previous_text = generated_text

            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "delta": delta_text,
                "text": generated_text,
                "logprobs": None,
                "finish_reason": "function_call" if func_call_found else None,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                },
            }

        if stop_found:
            break

    yield {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "delta": "",
        "text": generated_text,
        "logprobs": None,
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
    }
