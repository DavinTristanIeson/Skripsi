from typing import Annotated, Any, Callable, Optional, Sequence, TypeVar

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


def array_field_constraint(*, exact: Optional[int] = None, min: Optional[int] = None, max: Optional[int] = None):
  def validator(value: Sequence[Any], info: ValidationInfo):
    if exact is not None and len(value) != exact:
      raise ValueError(f"{info.field_name} should only have {exact} elements")
    if min is not None and len(value) < min:
      raise ValueError(f"{info.field_name} should have at least {min} elements")
    if max is not None and len(value) > max:
      raise ValueError(f"{info.field_name} can only have at most {max} elements")
    return value
  return validator     

__all__ = [
  "FilenameField",
  "DiscriminatedUnionValidator",
  "validate_http_url",
  "array_field_constraint"
]