from typing import Any, Callable

import pydantic


def fix_discriminated_union_loc(v: Any, handler: Callable[[Any], None]):
  try:
    return handler(v)
  except pydantic.ValidationError as e:
    raise e
  
"""Fixes pydantic discriminated union type being included in the ``loc`` field during validation."""
DiscriminatedUnionValidator = pydantic.WrapValidator(fix_discriminated_union_loc)
