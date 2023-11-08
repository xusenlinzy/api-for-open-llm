from enum import Enum

from pydantic import BaseModel


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int
