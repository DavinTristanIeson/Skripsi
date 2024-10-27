from typing import Annotated, Any, Callable

from pydantic import ConfigDict, ValidationError, WrapValidator
import pydantic

CommonModelConfig = ConfigDict(use_enum_values=True)

def fix_discriminated_union_loc(v: Any, handler: Callable[[Any], None]):
  try:
    return handler(v)
  except ValidationError as e:
    raise e
DiscriminatedUnionValidator = WrapValidator(fix_discriminated_union_loc)

FilenameField = Annotated[str, pydantic.Field(pattern=r"^[a-zA-Z0-9-_. ]+$", max_length=255)]

__all__ = [
  "DiscriminatedUnionValidator",
  "FilenameField"
]