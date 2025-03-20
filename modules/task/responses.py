import datetime
from enum import Enum
from typing import Generic, Optional, TypeVar
import pydantic

from modules.api import ExposedEnum

class TaskResponseType(Enum):
  Empty = "empty"

class TaskStatusEnum(str, Enum):
  Idle = "idle"
  Pending = "pending"
  Success = "success"
  Failed = "failed"

ExposedEnum().register(TaskStatusEnum)


T = TypeVar("T")
class TaskLog(pydantic.BaseModel):
  status: TaskStatusEnum
  message: str
  timestamp: datetime.datetime = pydantic.Field(
    default_factory=lambda: datetime.datetime.now()
  )
class TaskResponse(pydantic.BaseModel, Generic[T]):
  model_config = pydantic.ConfigDict(use_enum_values=True)
  
  id: str
  data: Optional[T]
  logs: list[TaskLog]
  status: TaskStatusEnum
  
__all__ = [
  "TaskLog",
  "TaskResponse",
  "TaskStatusEnum",
  "TaskResponseType"
]