""" this file is overdated and will be used """

import os
import sys
from typing import List, Optional, Any, Dict, Tuple

import torch
from loguru import logger
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from transformers.utils.versions import require_version

if sys.version_info >= (3, 9):
    from functools import cache
else:
    from functools import lru_cache as cache


class BaseModelAdapter:
    """ The base and default model adapter. """

    model_names = []

    def match(self, model_name) -> bool:
        """
        Check if the given model name matches any of the predefined model names.

        Args:
            model_name (str): The model name to check.

        Returns:
            bool: True if the model name matches any of the predefined model names, False otherwise.
        """

        return any(m in model_name for m in self.model_names) if self.model_names else True

    def load_model(
        self,
        model_name_or_path: Optional[str] = None,
        adapter_model: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a model and tokenizer based on the provided model name or path.

        Args:
            model_name_or_path (str, optional): The name or path of the model. Defaults to None.
            adapter_model (str, optional): The adapter model to load the tokenizer from. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizer]: A tuple containing the loaded model and tokenizer.
        """

        model_name_or_path = model_name_or_path or self.default_model_name_or_path
        tokenizer_kwargs = {"trust_remote_code": True, "use_fast": False}
        tokenizer_kwargs.update(self.tokenizer_kwargs)
        
        # load a tokenizer from adapter model if it exists.
        if adapter_model is not None:
            try:
                tokenizer = self.tokenizer_class.from_pretrained(
                    adapter_model, **tokenizer_kwargs,
                    )
            except OSError:
                tokenizer = self.tokenizer_class.from_pretrained(
                    model_name_or_path, **tokenizer_kwargs,
                    )
        else:
            tokenizer = self.tokenizer_class.from_pretrained(
                model_name_or_path, **tokenizer_kwargs,
                )

        config_kwargs = self.model_kwargs
        device = kwargs.get("device", "cuda")
        num_gpus = kwargs.get("num_gpus", 1)
        dtype = kwargs.get("dtype", "half")
        if device == "cuda":
            if "torch_dtype" not in config_kwargs:
                if dtype == "half":
                    config_kwargs["torch_dtype"] = torch.float16
                elif dtype == "bfloat16":
                    config_kwargs["torch_dtype"] = torch.bfloat16
                elif dtype == "float32":
                    config_kwargs["torch_dtype"] = torch.float32

            if num_gpus != 1:
                config_kwargs["device_map"] = "auto"
                # model_kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes

        # Quantization configurations (using bitsandbytes library).
        if kwargs.get("load_in_8bit", False):
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")

            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            config_kwargs["device_map"] = "auto" if device == "cuda" else None

            logger.info("Quantizing model to 8 bit.")

        elif kwargs.get("load_in_4bit", False):
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            require_version("peft>=0.4.0.dev0", "To fix: pip install git+https://github.com/huggingface/peft.git")

            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            config_kwargs["device_map"] = "auto" if device == "cuda" else None

            logger.info("Quantizing model to 4 bit.")

        if kwargs.get("device_map", None) == "auto":
            config_kwargs["device_map"] = "auto"

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        # Fix config (for Qwen)
        if hasattr(config, "fp16") and hasattr(config, "bf16"):
            setattr(config, "fp16", dtype == "half")
            setattr(config, "bf16", dtype == "bfloat16")
            config_kwargs.pop("torch_dtype", None)

        if kwargs.get("using_ptuning_v2", False) and adapter_model:
            config.pre_seq_len = kwargs.get("pre_seq_len", 128)

        # Load and prepare pretrained models (without valuehead).
        model = self.model_class.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            **config_kwargs
        )

        if device == "cpu":
            model = model.float()

        # post process for special tokens
        tokenizer = self.post_tokenizer(tokenizer)
        is_chatglm = "chatglm" in str(type(model))

        if adapter_model is not None:
            model = self.load_adapter_model(model, tokenizer, adapter_model, is_chatglm, config_kwargs, **kwargs)

        if is_chatglm or "baichuan" in str(type(model)) or "xverse" in str(type(model)):
            quantize = kwargs.get("quantize", None)
            if quantize and quantize != 16:
                logger.info(f"Quantizing model to {quantize} bit.")
                model = model.quantize(quantize)

        if device == "cuda" and num_gpus == 1 and "device_map" not in config_kwargs:
            model.to(device)

        # inference mode
        model.eval()

        return model, tokenizer

    def load_lora_model(
        self, model: PreTrainedModel, adapter_model: str, model_kwargs: Dict,
    ) -> PeftModel:
        """
        Load a LoRA model.

        This function loads a LoRA model using the specified pretrained model and adapter model.

        Args:
            model (PreTrainedModel): The base pretrained model.
            adapter_model (str): The name or path of the adapter model.
            model_kwargs (dict): Additional keyword arguments for the model.

        Returns:
            PeftModel: The loaded LoRA model.
        """
        return PeftModel.from_pretrained(
            model,
            adapter_model,
            torch_dtype=model_kwargs.get("torch_dtype", torch.float16),
        )

    def load_adapter_model(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        adapter_model: str, 
        is_chatglm: bool,
        model_kwargs: Dict,
        **kwargs: Any,
    ) -> PreTrainedModel:
        using_ptuning_v2 = kwargs.get("using_ptuning_v2", False)
        resize_embeddings = kwargs.get("resize_embeddings", False)
        if adapter_model and resize_embeddings and not is_chatglm:
            model_vocab_size = model.get_input_embeddings().weight.size(0)
            tokenzier_vocab_size = len(tokenizer)
            logger.info(f"Vocab of the base model: {model_vocab_size}")
            logger.info(f"Vocab of the tokenizer: {tokenzier_vocab_size}")

            if model_vocab_size != tokenzier_vocab_size:
                assert tokenzier_vocab_size > model_vocab_size
                logger.info("Resize model embeddings to fit tokenizer")
                model.resize_token_embeddings(tokenzier_vocab_size)

        if using_ptuning_v2:
            prefix_state_dict = torch.load(os.path.join(adapter_model, "pytorch_model.bin"))
            new_prefix_state_dict = {
                k[len("transformer.prefix_encoder."):]: v
                for k, v in prefix_state_dict.items()
                if k.startswith("transformer.prefix_encoder.")
            }
            model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
            model.transformer.prefix_encoder.float()
        else:
            model = self.load_lora_model(model, adapter_model, model_kwargs)

        return model

    def post_tokenizer(self, tokenizer) -> PreTrainedTokenizer:
        return tokenizer

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
        return {}

    @property
    def default_model_name_or_path(self):
        return "zpn/llama-7b"


# A global registry for all model adapters
model_adapters: List[BaseModelAdapter] = []


def register_model_adapter(cls):
    """ Register a model adapter. """
    model_adapters.append(cls())


@cache
def get_model_adapter(model_name: str) -> BaseModelAdapter:
    """
    Get a model adapter for a given model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        ModelAdapter: The model adapter that matches the given model name.
    """
    for adapter in model_adapters:
        if adapter.match(model_name):
            return adapter
    raise ValueError(f"No valid model adapter for {model_name}")


def load_model_and_tokenizer_old(
    model_name: str,
    model_name_or_path: Optional[str] = None,
    adapter_model: Optional[str] = None,
    quantize: Optional[int] = 16,
    device: Optional[str] = "cuda",
    load_in_8bit: Optional[bool] = False,
    **kwargs: Any,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a pre-trained model and tokenizer.

    Args:
        model_name (str): The name of the model.
        model_name_or_path (Optional[str], optional): The path or name of the pre-trained model. Defaults to None.
        adapter_model (Optional[str], optional): The name of the adapter model. Defaults to None.
        quantize (Optional[int], optional): The quantization level. Defaults to 16.
        device (Optional[str], optional): The device to load the model on. Defaults to "cuda".
        load_in_8bit (Optional[bool], optional): Whether to load the model in 8-bit mode. Defaults to False.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: A tuple containing the loaded model and tokenizer.
    """
    model_name = model_name.lower()

    if "tiger" in model_name:
        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

    # get model adapter
    adapter = get_model_adapter(model_name)
    model, tokenizer = adapter.load_model(
        model_name_or_path,
        adapter_model,
        device=device,
        quantize=quantize,
        load_in_8bit=load_in_8bit,
        **kwargs
    )
    return model, tokenizer


class ChatglmModelAdapter(BaseModelAdapter):
    """ https://github.com/THUDM/ChatGLM-6B """

    model_names = ["chatglm"]

    @property
    def model_class(self):
        return AutoModel

    @property
    def default_model_name_or_path(self):
        return "THUDM/chatglm2-6b"


class Chatglm3ModelAdapter(ChatglmModelAdapter):
    """ https://github.com/THUDM/ChatGLM-6B """

    model_names = ["chatglm3"]

    @property
    def tokenizer_kwargs(self):
        return {"encode_special_tokens": True}

    @property
    def default_model_name_or_path(self):
        return "THUDM/chatglm3-6b"


class LlamaModelAdapter(BaseModelAdapter):
    """ https://github.com/project-baize/baize-chatbot """

    model_names = ["alpaca", "baize", "openbuddy-llama", "ziya-llama", "guanaco", "llama2"]

    def post_tokenizer(self, tokenizer):
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.unk_token = "<unk>"
        return tokenizer

    @property
    def model_kwargs(self):
        return {"low_cpu_mem_usage": True}


class MossModelAdapter(BaseModelAdapter):
    """ https://github.com/OpenLMLab/MOSS """

    model_names = ["moss"]

    @property
    def default_model_name_or_path(self):
        return "fnlp/moss-moon-003-sft-int4"


class PhoenixModelAdapter(BaseModelAdapter):
    """ https://github.com/FreedomIntelligence/LLMZoo """

    model_names = ["phoenix"]

    @property
    def model_kwargs(self):
        return {"low_cpu_mem_usage": True}

    @property
    def tokenizer_kwargs(self):
        return {"use_fast": True}

    @property
    def default_model_name_or_path(self):
        return "FreedomIntelligence/phoenix-inst-chat-7b"


class FireflyModelAdapter(BaseModelAdapter):
    """ https://github.com/yangjianxin1/Firefly """

    model_names = ["firefly"]

    @property
    def model_kwargs(self):
        return {"torch_dtype": torch.float32}

    @property
    def tokenizer_kwargs(self):
        return {"use_fast": True}

    @property
    def default_model_name_or_path(self):
        return "YeungNLP/firefly-2b6"


class YuLanChatModelAdapter(BaseModelAdapter):
    """ https://github.com/RUC-GSAI/YuLan-Chat """

    model_names = ["yulan"]

    def post_tokenizer(self, tokenizer):
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.unk_token = "<unk>"
        return tokenizer

    @property
    def model_kwargs(self):
        return {"low_cpu_mem_usage": True}

    def load_adapter_model(self, model, tokenizer, adapter_model, is_chatglm, model_kwargs, **kwargs):
        adapter_model = AutoModelForCausalLM.from_pretrained(
            adapter_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        if model.model.embed_tokens.weight.size(0) + 1 == adapter_model.model.embed_tokens.weight.size(0):
            model.resize_token_embeddings(len(tokenizer))
            model.model.embed_tokens.weight.data[-1, :] = 0

        logger.info("Applying the delta")
        for name, param in tqdm(model.state_dict().items(), desc="Applying delta"):
            assert name in model.state_dict()
            param.data += model.state_dict()[name]

        return model


class TigerBotModelAdapter(BaseModelAdapter):
    """ https://github.com/TigerResearch/TigerBot """

    model_names = ["tiger"]

    @property
    def tokenizer_kwargs(self):
        return {"use_fast": True}

    @property
    def default_model_name_or_path(self):
        return "TigerResearch/tigerbot-7b-sft"


class OpenBuddyFalconModelAdapter(BaseModelAdapter):
    """ https://github.com/OpenBuddy/OpenBuddy """

    model_names = ["openbuddy-falcon"]

    @property
    def default_model_name_or_path(self):
        return "OpenBuddy/openbuddy-falcon-7b-v5-fp16"


class AnimaModelAdapter(LlamaModelAdapter):

    model_names = ["anima"]

    def load_lora_model(self, model, adapter_model, model_kwargs):
        return PeftModel.from_pretrained(model, adapter_model)


class BaiChuanModelAdapter(BaseModelAdapter):
    """ https://github.com/baichuan-inc/Baichuan-13B """

    model_names = ["baichuan"]

    def load_lora_model(self, model, adapter_model, model_kwargs):
        return PeftModel.from_pretrained(model, adapter_model)

    @property
    def default_model_name_or_path(self):
        return "baichuan-inc/Baichuan-13B-Chat"


class InternLMModelAdapter(BaseModelAdapter):
    """ https://github.com/InternLM/InternLM """

    model_names = ["internlm"]

    @property
    def default_model_name_or_path(self):
        return "internlm/internlm-chat-7b"


class StarCodeModelAdapter(BaseModelAdapter):
    """ https://github.com/bigcode-project/starcoder """

    model_names = ["starcode", "starchat"]

    @property
    def tokenizer_kwargs(self):
        return {}

    @property
    def default_model_name_or_path(self):
        return "HuggingFaceH4/starchat-beta"


class AquilaModelAdapter(BaseModelAdapter):
    """ https://github.com/FlagAI-Open/FlagAI """

    model_names = ["aquila"]

    @property
    def default_model_name_or_path(self):
        return "BAAI/AquilaChat-7B"


class QwenModelAdapter(BaseModelAdapter):
    """ https://github.com/QwenLM/Qwen-7B """

    model_names = ["qwen"]

    @property
    def default_model_name_or_path(self):
        return "Qwen/Qwen-7B-Chat"


class XverseModelAdapter(BaseModelAdapter):
    """ https://github.com/xverse-ai/XVERSE-13B """

    model_names = ["xverse"]

    @property
    def default_model_name_or_path(self):
        return "xverse/XVERSE-13B-Chat"


class CodeLlamaModelAdapter(LlamaModelAdapter):
    """ https://github.com/project-baize/baize-chatbot """

    model_names = ["code-llama"]

    @property
    def tokenizer_class(self):
        require_version("transformers>=4.33.1", "To fix: pip install transformers>=4.33.1")
        from transformers import CodeLlamaTokenizer

        return CodeLlamaTokenizer

    @property
    def default_model_name_or_path(self):
        return "codellama/CodeLlama-7b-Instruct-hf"


register_model_adapter(ChatglmModelAdapter)
register_model_adapter(Chatglm3ModelAdapter)
register_model_adapter(LlamaModelAdapter)
register_model_adapter(MossModelAdapter)
register_model_adapter(PhoenixModelAdapter)
register_model_adapter(FireflyModelAdapter)
register_model_adapter(YuLanChatModelAdapter)
register_model_adapter(TigerBotModelAdapter)
register_model_adapter(OpenBuddyFalconModelAdapter)
register_model_adapter(AnimaModelAdapter)
register_model_adapter(BaiChuanModelAdapter)
register_model_adapter(InternLMModelAdapter)
register_model_adapter(AquilaModelAdapter)
register_model_adapter(QwenModelAdapter)
register_model_adapter(XverseModelAdapter)
register_model_adapter(CodeLlamaModelAdapter)

# After all adapters, try the default base adapter.
register_model_adapter(BaseModelAdapter)
