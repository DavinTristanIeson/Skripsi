import abc
from typing import Any, Generic, Optional, TypeVar, Union
import pydantic

T = TypeVar("T")

class ApiError(Exception):
  def __init__(self, message: str, status_code: int):
    self.message = message
    self.status_code = status_code

  def __str__(self):
    return self.message
  
class ApiErrorAdaptableException(abc.ABC, Exception):
  @abc.abstractmethod
  def to_api(self)->ApiError:
    ...

  def __str__(self):
    return self.to_api().message
    
class ApiResult(pydantic.BaseModel, Generic[T]):
  data: T
  message: Optional[str]
  def as_json(self):
    from fastapi.encoders import jsonable_encoder
    return jsonable_encoder(self)

class ApiErrorResult(pydantic.BaseModel):
  message: str
  errors: Optional[dict[Union[str, int], Any]] = None
  def as_json(self):
    from fastapi.encoders import jsonable_encoder
    return jsonable_encoder(self)
  

__all__ = [
  "ApiError",
  "ApiResult",
  "ApiErrorResult"
]