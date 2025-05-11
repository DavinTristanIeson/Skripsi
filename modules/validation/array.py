
from typing import Any, Optional, Sequence

from pydantic import AfterValidator, ValidationInfo

def array_field_constraint(*, exact: Optional[int] = None, min: Optional[int] = None, max: Optional[int] = None):
  def validator(value: Sequence[Any], info: ValidationInfo):
    if exact is not None and len(value) != exact:
      raise ValueError(f"{info.field_name} should only have {exact} elements")
    if min is not None and len(value) < min:
      raise ValueError(f"{info.field_name} should have at least {min} elements")
    if max is not None and len(value) > max:
      raise ValueError(f"{info.field_name} can only have at most {max} elements")
    return value
  return AfterValidator(validator)

def __validate_start_le_end(value: Optional[tuple[Any, Any]], info: ValidationInfo):
  if value is None:
    return value
  if len(value) != 2:
    raise ValueError(f"{info.field_name} should contain at least two elements, but received {value} instead.")
  if value[0] > value[1]:
    return (value[1], value[0])
  else:
    return value

StartBeforeEndValidator = AfterValidator(__validate_start_le_end)


__all__ = [
  "array_field_constraint",
  "StartBeforeEndValidator",
]  
