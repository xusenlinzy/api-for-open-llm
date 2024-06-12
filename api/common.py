from __future__ import annotations

import os
from typing import Any, Dict, Type

import pydantic
from pydantic import BaseModel

PYDANTIC_V2 = pydantic.VERSION.startswith("2.")


def dictify(data: "BaseModel", **kwargs) -> Dict[str, Any]:
    try:  # pydantic v2
        return data.model_dump(**kwargs)
    except AttributeError:  # pydantic v1
        return data.dict(**kwargs)


def jsonify(data: "BaseModel", **kwargs) -> str:
    try:  # pydantic v2
        return data.model_dump_json(**kwargs)
    except AttributeError:  # pydantic v1
        return data.json(**kwargs)


def model_validate(data: Type["BaseModel"], obj: Any) -> "BaseModel":
    try:  # pydantic v2
        return data.model_validate(obj)
    except AttributeError:  # pydantic v1
        return data.parse_obj(obj)


def disable_warnings(model: Type["BaseModel"]):
    # Disable warning for model_name settings
    if PYDANTIC_V2:
        model.model_config["protected_namespaces"] = ()


def get_bool_env(key, default="false"):
    return os.environ.get(key, default).lower() == "true"


def get_env(key, default):
    val = os.environ.get(key, "")
    return val or default
