import torch
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModel,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    BloomTokenizerFast,
    BloomForCausalLM,
)


def load_chatglm_tokenizer_and_model(base_model, adapter_model=None, quantize=16, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        base_model,
        trust_remote_code=True
    )

    if adapter_model:
        model = PeftModel.from_pretrained(
            model,
            adapter_model,
            torch_dtype=torch.float32,
        )

    if device == "cpu":
        model.float()

    if device != "cpu":
        model = model.half()
        if quantize != 16 and adapter_model is None:
            model = model.quantize(quantize)

    model.to(device)
    model.eval()

    return tokenizer, model


def load_llama_tokenizer_and_model(base_model, adapter_model=None, load_8bit=False, device="cuda:0"):
    if adapter_model:
        try:
            tokenizer = LlamaTokenizer.from_pretrained(adapter_model)
        except:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    if adapter_model:
        model_vocab_size = model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")

        if model_vocab_size != tokenzier_vocab_size:
            assert tokenzier_vocab_size > model_vocab_size
            print("Resize model embeddings to fit tokenizer")
            model.resize_token_embeddings(tokenzier_vocab_size)

        model = PeftModel.from_pretrained(
            model,
            adapter_model,
            torch_dtype=torch.float16,
        )

    if device == "cpu":
        model.float()

    if not load_8bit and device != "cpu":
        model.half()  # seems to fix bugs for some users.

    model.to(device)
    model.eval()

    return tokenizer, model


def load_auto_tokenizer_and_model(
    model_name,
    base_model,
    adapter_model=None,
    quantize=16,
    device="cuda:0",
    load_8bit=False,
):
    if "chatglm" in model_name.lower():
        tokenizer, model = load_chatglm_tokenizer_and_model(
            base_model,
            adapter_model=adapter_model,
            quantize=quantize,
            device=device,
        )
    elif "llama" in model_name.lower():
        tokenizer, model = load_llama_tokenizer_and_model(
            base_model,
            adapter_model=adapter_model,
            load_8bit=load_8bit,
            device=device,
        )
    elif "moss" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True).half().to(device)
        model.eval()
    elif "phoenix" in model_name.lower():
        tokenizer = BloomTokenizerFast.from_pretrained(base_model)
        model = BloomForCausalLM.from_pretrained(
            base_model,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        ).to(device)
        model.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
        model.eval()

    return tokenizer, model
