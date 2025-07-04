"""
As this module will include FastAPI dependency, it should be manually imported and not from modules.api.
"""
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from modules.logger import ProvisionedLogger
import traceback

from .wrapper import ApiError, ApiErrorAdaptableException, ApiErrorResult

logger = ProvisionedLogger().provision("FastAPI Error Handler")

def api_error_exception_handler(request: Request, exc: ApiError):
  logger.error(f"API Error while handling {request.url}. Error: {''.join(traceback.format_exception(exc))}")
  return JSONResponse(content=ApiErrorResult(message=exc.message).model_dump(), status_code=exc.status_code)

def api_error_adaptable_exception_handler(request: Request, exc: ApiErrorAdaptableException):
  logger.error(f"API Error while handling {request.url}. Error: {''.join(traceback.format_exception(exc))}")
  api_error = exc.to_api()
  return JSONResponse(content=ApiErrorResult(message=api_error.message).model_dump(), status_code=api_error.status_code)


def http_exception_handler(request: Request, exc: HTTPException):
  logger.error(f"API Error while handling {request.url}. Error: {''.join(traceback.format_exception(exc))}")
  logger.exception(exc)
  return JSONResponse(content=ApiErrorResult(message=exc.detail).model_dump(), status_code=exc.status_code)

def default_exception_handler(request: Request, exc: Exception):
  logger.error(f"Error while handling {request.url}. Error: {''.join(traceback.format_exception(exc))}")
  logger.exception(exc)
  return JSONResponse(content=ApiErrorResult(message="An unexpected error has occurred in the server.").model_dump(), status_code=500)

async def validation_exception_handler(request: Request, exc: RequestValidationError):
  raw_errors = list(exc.errors())
  errors = {}
  for error in raw_errors:
    if error["type"] == "json_invalid":
      return JSONResponse(
        status_code=400,
        content=ApiErrorResult(
          message="Invalid JSON in body",
          errors=None
        ).as_json(),
      )

    error_mapper = errors
    error_path = error['loc']
    # Create error tree
    for idx, loc in enumerate(error_path[1:-1]):
      if idx == 0:
        continue
      # Nest into the error tree
      if loc not in error_mapper:
        error_mapper[loc] = {}
      error_mapper = error_mapper[loc]

    error_mapper[error_path[-1]] = str(error['msg'])

  message = str(raw_errors[0]['msg'])
  logger.error(f"Error while handling {request.url}. Error: {exc}")
  return JSONResponse(
    status_code=422,
    content=ApiErrorResult(message=message, errors=errors).as_json(),
  )

def register_error_handlers(app: FastAPI):
  app.exception_handler(ApiError)(
    api_error_exception_handler
  )
  app.exception_handler(ApiErrorAdaptableException)(
    api_error_adaptable_exception_handler
  )
  app.exception_handler(RequestValidationError)(
    validation_exception_handler
  )
  app.exception_handler(HTTPException)(
    http_exception_handler
  )
  app.exception_handler(Exception)(
    default_exception_handler
  )


__all__ = [
  "register_error_handlers"
]