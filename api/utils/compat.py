from __future__ import annotations

from typing import Any, TypeVar, cast, Dict

import pydantic

_ModelT = TypeVar("_ModelT", bound=pydantic.BaseModel)

# --------------- Pydantic v2 compatibility ---------------

PYDANTIC_V2 = pydantic.VERSION.startswith("2.")


def model_copy(model: _ModelT) -> _ModelT:
    if PYDANTIC_V2:
        return model.model_copy()
    return model.copy()  # type: ignore


def model_json(model: pydantic.BaseModel, **kwargs) -> str:
    if PYDANTIC_V2:
        return model.model_dump_json(**kwargs)
    return model.json(**kwargs)  # type: ignore


def model_dump(model: pydantic.BaseModel, **kwargs) -> Dict[str, Any]:
    if PYDANTIC_V2:
        return model.model_dump(**kwargs)
    return cast(
        "dict[str, Any]",
        model.dict(**kwargs),
    )


def model_parse(model: _ModelT, data: Any) -> _ModelT:
    if PYDANTIC_V2:
        return model.model_validate(data)
    return model.parse_obj(data)  # pyright: ignore[reportDeprecated]


def disable_warnings(model: pydantic.BaseModel):
    # Disable warning for model_name settings
    if PYDANTIC_V2:
        model.model_config["protected_namespaces"] = ()
