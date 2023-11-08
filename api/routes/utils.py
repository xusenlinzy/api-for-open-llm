from typing import Optional, Union

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from openai.types.chat import CompletionCreateParams as ChatCompletionCreateParams
from openai.types.completion_create_params import CompletionCreateParams

from api.config import config
from api.utils.constants import ErrorCode
from api.utils.protocol import ErrorResponse


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
):
    if config.API_KEYS is not None:
        if auth is None or (token := auth.credentials) not in config.API_KEYS:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message, code=code).dict(), status_code=500)


def check_requests(
    request: Union[CompletionCreateParams, ChatCompletionCreateParams]
) -> Optional[JSONResponse]:
    # Check all params
    if request.get("max_tokens") is not None and request.get("max_tokens") <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.get('max_tokens')} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.get("n") is not None and request.get("n") <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.get('n')} is less than the minimum of 1 - 'n'",
        )
    if request.get("temperature") is not None and request.get("temperature") < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.get('temperature')} is less than the minimum of 0 - 'temperature'",
        )
    if request.get("temperature") is not None and request.get("temperature") > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.get('temperature')} is greater than the maximum of 2 - 'temperature'",
        )
    if request.get("top_p") is not None and request.get("top_p") < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.get('top_p')} is less than the minimum of 0 - 'top_p'",
        )
    if request.get("top_p") is not None and request.get("top_p") > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.get('top_p')} is greater than the maximum of 1 - 'temperature'",
        )
    if request.get("stop") is not None and (
            not isinstance(request.get("stop"), str) and not isinstance(request.get("stop"), list)
    ):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.get('stop')} is not valid under any of the given schemas - 'stop'",
        )

    return None
