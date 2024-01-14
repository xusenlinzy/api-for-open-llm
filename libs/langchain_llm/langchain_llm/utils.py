from typing import Optional

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def apply_lora(
    base_model_path: str,
    lora_path: str,
    target_model_path: str,
    max_shard_size: Optional[str] = "2GB",
    safe_serialization: Optional[bool] = True,
):

    logger.info(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        trust_remote_code=True,
    )

    logger.info(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(base, lora_path)

    logger.info("Applying the LoRA")
    model = lora_model.merge_and_unload()

    logger.info(f"Saving the target model to {target_model_path}")
    model.save_pretrained(
        target_model_path,
        max_shard_size=max_shard_size,
        safe_serialization=safe_serialization,
    )
    base_tokenizer.save_pretrained(target_model_path)
