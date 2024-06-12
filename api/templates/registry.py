from collections import OrderedDict
from functools import lru_cache
from typing import Optional, TYPE_CHECKING

from api.templates.base import ChatTemplate

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


TEMPLATES = OrderedDict({"default": ChatTemplate})


def register_template(name):
    """ Register a chat template. """

    def wrapper(cls):
        TEMPLATES[name] = cls
        return cls

    return wrapper


@lru_cache
def get_template(
    name: str,
    tokenizer: "PreTrainedTokenizer",
    model_max_length: Optional[int] = 8192,
) -> ChatTemplate:
    """ Get a chat template for a template name. """
    cls = TEMPLATES.get(name, ChatTemplate)
    return cls(tokenizer=tokenizer, model_max_length=model_max_length)
