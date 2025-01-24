# Common resources

from enum import Enum
from typing import Generic, Optional, TypeVar

import pydantic
from common.models.enum import ExposedEnum


class TaskStatus(str, Enum):
  Idle = "idle"
  Pending = "pending"
  Success = "success"
  Failed = "failed"

ExposedEnum().register(TaskStatus)

T = TypeVar("T")
class TaskResult(pydantic.BaseModel, Generic[T]):
  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, use_enum_values=True)

  data: T
  status: TaskStatus
  message: Optional[str]
  progress: Optional[float]
  error: Optional[str]