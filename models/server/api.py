
from typing import Any, Generic, Optional, TypeVar
from fastapi.encoders import jsonable_encoder
import pydantic

T = TypeVar("T")

class ApiResult(pydantic.BaseModel, Generic[T]):
  data: T
  message: Optional[str]
  def as_json(self):
    return jsonable_encoder(self)

class ApiErrorResult(pydantic.BaseModel):
  message: str
  errors: Optional[dict[str, Any]] = None