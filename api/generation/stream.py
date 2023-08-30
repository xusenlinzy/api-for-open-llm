import gc
from threading import Thread
from types import MethodType
from typing import Iterable

import torch
from transformers import TextIteratorStreamer, PreTrainedModel

from api.generation.baichuan import check_is_baichuan, build_baichuan_chat_input
from api.generation.qwen import check_is_qwen, build_qwen_chat_input
from api.generation.utils import prepare_logits_processor, is_partial_stop, apply_stopping_strings
from api.generation.xverse import check_is_xverse, build_xverse_chat_input


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
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)

    stop_token_ids = params.get("stop_token_ids", None) or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    infilling = params.get("infilling", False)
    suffix_first = params.get("suffix_first", False)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    if isinstance(prompt, list) and check_is_baichuan(model):
        input_ids = build_baichuan_chat_input(tokenizer, prompt, context_len, max_new_tokens)
    elif isinstance(prompt, list) and check_is_qwen(model):
        input_ids = build_qwen_chat_input(tokenizer, prompt, context_len, max_new_tokens)
        stop_token_ids.extend([tokenizer.im_end_id, tokenizer.im_start_id])
    elif isinstance(prompt, list) and check_is_xverse(model):
        input_ids = build_xverse_chat_input(tokenizer, prompt, context_len, max_new_tokens)
    else:
        if infilling:
            input_ids = tokenizer(prompt, suffix_first=suffix_first).input_ids
            stop_token_ids.append(tokenizer.eot_id)
        else:
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
                rfind_start = len(prompt) if isinstance(prompt, str) else 0
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=False if check_is_qwen(model) else True,  # fix for qwen react
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
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_stream_v2(
    model,
    tokenizer,
    params,
    device: str,
    context_len: int,
):
    prompt = params["prompt"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", 40))
    max_new_tokens = int(params.get("max_tokens", 256))

    stop_token_ids = params.get("stop_token_ids", None) or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    stop_strings = params.get("stop", [])
    infilling = params.get("infilling", False)
    suffix_first = params.get("suffix_first", False)

    if isinstance(prompt, list) and check_is_baichuan(model):
        input_ids = build_baichuan_chat_input(tokenizer, prompt, context_len, max_new_tokens)
    elif isinstance(prompt, list) and check_is_qwen(model):
        input_ids = build_qwen_chat_input(tokenizer, prompt, context_len, max_new_tokens)
    elif isinstance(prompt, list) and check_is_xverse(model):
        input_ids = build_xverse_chat_input(tokenizer, prompt, context_len, max_new_tokens)
    else:
        if infilling:
            input_ids = tokenizer(prompt, suffix_first=suffix_first).input_ids
        else:
            input_ids = tokenizer(prompt).input_ids

        if model.config.is_encoder_decoder:
            max_src_len = context_len
        else:  # truncate
            max_src_len = context_len - max_new_tokens - 1
        input_ids = input_ids[-max_src_len:]

    input_echo_len = len(input_ids)

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

    generated_text = ""
    for i, new_text in enumerate(streamer):
        generated_text += new_text
        generated_text, stop_found = apply_stopping_strings(generated_text, stop_strings)

        yield {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": None,
            "error_code": 0,
        }

        if stop_found:
            break

    if stop_found:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": generated_text,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
        "error_code": 0,
    }
