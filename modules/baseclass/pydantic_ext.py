import pydantic

class PydanticBaseModel(pydantic.BaseModel):
  """Use this as BaseModel rather than pydantic.BaseModel"""
  model_config = pydantic.ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)

__all__ = [
  "PydanticBaseModel",
]