from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from common.models.api import ApiError
from common.models.api import ApiErrorResult



def default_exception_handler(request: Request, exc: ApiError):
  return JSONResponse(content=ApiErrorResult(message=exc.message).model_dump(), status_code=exc.status_code)

async def validation_exception_handler(request: Request, exc: RequestValidationError):
  raw_errors = list(exc.errors())
  errors = {}
  for error in raw_errors:
    error_mapper = errors
    error_path = error['loc']
    # Create error tree
    for idx, loc in error_path[1:-1]:
      if idx == 0:
        continue
      # Nest into the error tree
      if loc not in error_mapper:
        error_mapper[loc] = {}
      error_mapper = error_mapper[loc]

    error_mapper[error_path[-1]] = error['msg']

  message = str(raw_errors[0]['msg'])
  return JSONResponse(
    status_code=400,
    content=ApiErrorResult(message=message, errors=errors).model_dump(),
  )

def register_error_handlers(app: FastAPI):
  app.exception_handler(ApiError)(
    default_exception_handler
  )
  app.exception_handler(RequestValidationError)(
    validation_exception_handler
  )

__all__ = [
  "register_error_handlers"
]