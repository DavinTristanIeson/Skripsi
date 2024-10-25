from typing import Any, Callable

from pydantic import ConfigDict, ValidationError, WrapValidator

CommonModelConfig = ConfigDict(use_enum_values=True)

def fix_discriminated_union_loc(v: Any, handler: Callable[[Any], None]):
  try:
    return handler(v)
  except ValidationError as e:
    raise e
DiscriminatedUnionValidator = WrapValidator(fix_discriminated_union_loc)

__all__ = [
  "DiscriminatedUnionValidator"
]