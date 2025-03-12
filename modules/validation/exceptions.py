from typing import Any, Callable

import pydantic
from pydantic_core import InitErrorDetails


def __fix_discriminated_union_loc(v: Any, handler: Callable[[Any], None], info: pydantic.ValidationInfo):
  try:
    return handler(v)
  except pydantic.ValidationError as exc:
    adjusted_errors: list[InitErrorDetails] = []
    for error in exc.errors():
      adjusted_error: InitErrorDetails = {
        "type": error["type"],
        "loc": error["loc"][1:],
        "input": error["input"],
      }
      if "ctx" in error:
        adjusted_error["ctx"] = error["ctx"]
      adjusted_errors.append(adjusted_error)
    
    raise pydantic.ValidationError.from_exception_data(title=exc.title, line_errors=adjusted_errors, input_type=info.mode) from None
  
"""Fixes pydantic discriminated union type being included in the ``loc`` field during validation."""
DiscriminatedUnionValidator = pydantic.WrapValidator(__fix_discriminated_union_loc)

__all__ = [
  "DiscriminatedUnionValidator"
]
