
from typing import Generic, Optional, TypeVar
import pydantic

# Resources
T = TypeVar("T")

# Schemas
class UserDataSchema(pydantic.BaseModel, Generic[T]):
  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
  name: str = pydantic.Field(min_length=1)
  tags: Optional[list[str]]
  description: Optional[str]
  data: T

# Resources
class UserDataResource(pydantic.BaseModel, Generic[T]):
  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
  id: str
  name: str
  tags: Optional[list[str]]
  description: Optional[str]
  data: T
  
  @staticmethod
  def from_schema(schema: UserDataSchema, id: str)->"UserDataResource[T]":
    return UserDataResource(
      id=id,
      name=schema.name,
      tags=schema.tags,
      description=schema.description,
      data=schema.data,
    )

__all__ = [
  "UserDataSchema",
  "UserDataResource",
]