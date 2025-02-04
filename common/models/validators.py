from typing import Annotated, Any, Callable, TypeVar

from pydantic import ConfigDict, ValidationError, ValidationInfo, WrapValidator, model_validator
import pydantic
from pydantic_core import InitErrorDetails

CommonModelConfig = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)

def fix_discriminated_union_loc(v: Any, handler: Callable[[Any], None]):
  try:
    return handler(v)
  except ValidationError as e:
    raise e
DiscriminatedUnionValidator = WrapValidator(fix_discriminated_union_loc)

FilenameField = Annotated[str, pydantic.Field(pattern=r"^[a-zA-Z0-9-_. ]+$", max_length=255)]
FilePathField = Annotated[str, pydantic.Field(pattern=r"^[a-zA-Z0-9-_. \/\\:]+$")]

class __LinkValidationModel(pydantic.BaseModel):
  url: pydantic.AnyHttpUrl
@staticmethod
def validate_http_url(url: str):
  if url is None or not isinstance(url, str) or len(url) == 0:
    return False
  try:
    __LinkValidationModel.model_validate(dict(url=url))
    return True
  except pydantic.ValidationError:
    return False

__all__ = [
  "FilenameField",
  "DiscriminatedUnionValidator",
  "validate_http_url",
]