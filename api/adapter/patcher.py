""" from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llmtuner/model/patcher.py """
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from types import MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
)

import torch
from loguru import logger
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig,
)
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_npu_available
)
from transformers.utils.versions import require_version

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer


_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
try:
    _is_bf16_available = is_torch_bf16_gpu_available()
except:
    _is_bf16_available = False


def is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def get_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except:
        return "0.0.0"


def is_flash_attn2_available():
    return is_package_available("flash_attn") and get_package_version("flash_attn").startswith("2")


def infer_optim_dtype(model_dtype: torch.dtype) -> torch.dtype:
    r"""
    Infers the optimal dtype according to the model_dtype and device compatibility.
    """
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32


def _configure_rope(config: "PretrainedConfig", rope_scaling: str = None) -> None:
    if not hasattr(config, "rope_scaling"):
        logger.warning("Current model does not support RoPE scaling.")
        return

    scaling_factor = 2.0
    setattr(config, "rope_scaling", {"type": rope_scaling, "factor": scaling_factor})
    logger.info(f"Using {rope_scaling} scaling strategy and setting scaling factor to {scaling_factor}.")


def _configure_flashattn(config_kwargs: Dict[str, Any]) -> None:
    if not is_flash_attn2_available():
        logger.warning("FlashAttention2 is not installed.")
        return

    config_kwargs["use_flash_attention_2"] = True
    logger.info("Using FlashAttention-2 for faster and inference.")


def _configure_quantization(
    config_kwargs: Dict[str, Any],
    load_in_8bits: bool = False,
    load_in_4bits: bool = False,
) -> None:

    if load_in_8bits:
        require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
        config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Quantizing model to 8 bit.")

    elif load_in_4bits:
        require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
        config_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config_kwargs.get("torch_dtype", torch.float16),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Quantizing model to 4 bit.")

    if load_in_8bits or load_in_4bits:
        config_kwargs["device_map"] = {"": get_current_device()}
    else:
        config_kwargs["device_map"] = get_current_device()


def patch_tokenizer(tokenizer: "PreTrainedTokenizer") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        logger.info(f"Add eos token: {tokenizer.eos_token}")

    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad token: {tokenizer.pad_token}")


def patch_config(
    config: "PretrainedConfig",
    config_kwargs: Dict[str, Any],
    compute_dtype: Optional[str] = None,
    **kwargs,
):
    if compute_dtype is None:  # priority: bf16 > fp16 > fp32
        compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))
    else:
        _DTYPE_MAP = {
            "half": torch.float16,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = _DTYPE_MAP.get(compute_dtype, torch.float16)

    config_kwargs["torch_dtype"] = compute_dtype

    if getattr(config, "model_type", None) == "qwen":
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, compute_dtype == dtype)

    rope_scaling = kwargs.get("rope_scaling", None)
    if rope_scaling is not None:
        _configure_rope(config, rope_scaling)

    if kwargs.get("flash_attn", False):
        _configure_flashattn(config_kwargs)

    _configure_quantization(
        config_kwargs,
        kwargs.get("load_in_8bit", False),
        kwargs.get("load_in_4bit", False),
    )


def patch_model(model: "PreTrainedModel") -> None:
    if model.config.model_type == "minicpmv":
        return
    if "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)


def get_current_device() -> torch.device:
    r"""
    Gets the current available device.
    """
    if is_torch_npu_available():
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)
