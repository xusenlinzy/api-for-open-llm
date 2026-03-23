from typing import Optional, List

from loguru import logger
from openai import OpenAI


# MiniMax supported models and their context lengths
MINIMAX_MODELS = {
    "MiniMax-M2.7": 1048576,
    "MiniMax-M2.7-highspeed": 1048576,
    "MiniMax-M2.5": 245760,
    "MiniMax-M2.5-highspeed": 204800,
}


class MiniMaxEngine:
    """Engine that proxies requests to MiniMax Cloud API via OpenAI-compatible interface."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        api_base: str = "https://api.minimax.io/v1",
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name

        logger.info(f"Using MiniMax Cloud API with model: {self.model_name}")
        logger.info(f"MiniMax API base: {api_base}")

    @staticmethod
    def clamp_temperature(temperature: Optional[float]) -> Optional[float]:
        """Clamp temperature to MiniMax accepted range [0, 1]."""
        if temperature is None:
            return None
        return max(0.0, min(1.0, temperature))

    @staticmethod
    def available_models() -> List[str]:
        """Return list of available MiniMax model names."""
        return list(MINIMAX_MODELS.keys())
