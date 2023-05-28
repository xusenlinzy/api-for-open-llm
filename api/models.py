from typing import Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# A global registry for all model adapters
model_adapters = []


def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())


def get_model_adapter(model_name: str):
    """Get a model adapter for a model_name."""
    for adapter in model_adapters:
        if adapter.match(model_name):
            return adapter
    raise ValueError(f"No valid model adapter for {model_name}")


class BaseModelAdapter:
    """The base and the default model adapter."""

    def match(self, model_name):
        return True

    def load_model(self, model_name_or_path: str, adapter_model: Optional[str] = None, **kwargs):
        """ Load model through transformers. """
        tokenizer_kwargs = self.tokenizer_kwargs
        if adapter_model is not None:
            try:
                tokenizer = self.tokenizer_class.from_pretrained(adapter_model, **tokenizer_kwargs)
            except OSError:
                tokenizer = self.tokenizer_class.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        else:
            tokenizer = self.tokenizer_class.from_pretrained(model_name_or_path, **tokenizer_kwargs)

        device = kwargs.get("device", "cuda:0")
        model_kwargs = self.model_kwargs
        if device == "cpu":
            model_kwargs["torch_dtype"] = torch.float32
        else:
            if "torch_dtype" not in model_kwargs:
                model_kwargs["torch_dtype"] = torch.float16

        model = self.model_class.from_pretrained(model_name_or_path, load_in_8bit=kwargs.get("load_in_8bit", False), **model_kwargs)
        is_chatglm = "chatglm" in str(type(model))

        if adapter_model is not None:
            model_vocab_size = model.get_input_embeddings().weight.size(0)
            tokenzier_vocab_size = len(tokenizer)
            print(f"Vocab of the base model: {model_vocab_size}")
            print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")

            if model_vocab_size != tokenzier_vocab_size:
                assert tokenzier_vocab_size > model_vocab_size
                print("Resize model embeddings to fit tokenizer")
                model.resize_token_embeddings(tokenzier_vocab_size)

            model = self.load_adapter_model(model, adapter_model, model_kwargs)

        if is_chatglm:
            quantize = kwargs.get("quantize", None)
            if quantize and quantize != 16:
                model = model.quantize(quantize)

        if device != "cpu":
            model.to(device)
            model.eval()

        return model, tokenizer

    def load_adapter_model(self, model, adapter_model, model_kwargs):
        return PeftModel.from_pretrained(
            model,
            adapter_model,
            torch_dtype=model_kwargs.get("torch_dtype", torch.float16),
        )

    @property
    def model_class(self):
        return AutoModelForCausalLM

    @property
    def model_kwargs(self):
        return {}

    @property
    def tokenizer_class(self):
        return AutoTokenizer

    @property
    def tokenizer_kwargs(self):
        return {"use_fast": False}


def load_model(
    model_name: str,
    model_name_or_path: str,
    adapter_model: Optional[str] = None,
    quantize: Optional[int] = 16,
    device: Optional[str] = "cuda:0",
    load_in_8bit: bool = False,
):
    model_name = model_name.lower()
    adapter = get_model_adapter(model_name)
    model, tokenizer = adapter.load_model(
        model_name_or_path,
        adapter_model,
        device=device,
        quantize=quantize,
        load_in_8bit=load_in_8bit,
    )
    return model, tokenizer


class ChatglmModelAdapter(BaseModelAdapter):

    def match(self, model_name):
        return "chatglm" in model_name

    @property
    def model_class(self):
        return AutoModel

    @property
    def model_kwargs(self):
        return {"trust_remote_code": True}

    @property
    def tokenizer_kwargs(self):
        return {"use_fast": False, "trust_remote_code": True}


class LlamaModelAdapter(BaseModelAdapter):

    def match(self, model_name):
        return "alpaca" in model_name or "baize" in model_name

    @property
    def model_kwargs(self):
        return {"low_cpu_mem_usage": True}


class MossModelAdapter(BaseModelAdapter):

    def match(self, model_name):
        return "moss" in model_name

    @property
    def model_kwargs(self):
        return {"trust_remote_code": True}

    @property
    def tokenizer_kwargs(self):
        return {"use_fast": False, "trust_remote_code": True}


class PhoenixModelAdapter(BaseModelAdapter):

    def match(self, model_name):
        return "phoenix" in model_name

    @property
    def model_kwargs(self):
        return {"low_cpu_mem_usage": True}

    @property
    def tokenizer_kwargs(self):
        return {"use_fast": True}


class FireflyModelAdapter(BaseModelAdapter):

    def match(self, model_name):
        return "firefly" in model_name

    @property
    def model_kwargs(self):
        return {"torch_dtype": torch.float32}

    @property
    def tokenizer_kwargs(self):
        return {"use_fast": True}


register_model_adapter(ChatglmModelAdapter)
register_model_adapter(LlamaModelAdapter)
register_model_adapter(MossModelAdapter)
register_model_adapter(PhoenixModelAdapter)
register_model_adapter(FireflyModelAdapter)

register_model_adapter(BaseModelAdapter)
