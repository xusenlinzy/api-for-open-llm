from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Optional,
    Tuple,
    Any,
)

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .patcher import (
    patch_config,
    patch_tokenizer,
    patch_model,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def load_model_and_tokenizer(
    model_name_or_path: str,
    use_fast_tokenizer: Optional[bool] = False,
    dtype: Optional[str] = None,
    device_map: Optional[Any] = None,
    load_in_8bit: Optional[bool] = False,
    load_in_4bit: Optional[bool] = False,
    rope_scaling: Optional[str] = None,
    flash_attn: Optional[bool] = False,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support inference.
    """
    config_kwargs = {"trust_remote_code": True}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=True,
    )
    patch_tokenizer(tokenizer)

    config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
    patch_config(
        config,
        config_kwargs,
        dtype,
        rope_scaling=rope_scaling,
        flash_attn=flash_attn,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    if device_map:
        config_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
        **config_kwargs
    )

    patch_model(model)
    model.eval()

    return model, tokenizer
