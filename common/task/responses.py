import datetime
from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Any, Literal, Optional, Union
import pydantic

from common.models.enum import ExposedEnum

class TaskResponseType(Enum):
  Empty = "empty"

class TaskStatus(str, Enum):
  Idle = "idle"
  Pending = "pending"
  Success = "success"
  Failed = "failed"
ExposedEnum().register(TaskStatus)

class TaskResponseData(SimpleNamespace):
  class Empty(pydantic.BaseModel):
    type: Literal[TaskResponseType.Empty] = TaskResponseType.Empty

  TypeUnion = Union[Empty]
  DiscriminatedUnion = Annotated[TypeUnion, pydantic.Field(discriminator="type")]

class TaskResponse(pydantic.BaseModel):
  id: str
  data: TaskResponseData.DiscriminatedUnion
  status: TaskStatus
  message: Optional[str] = None
  timestamp: datetime.datetime = pydantic.Field(
    default_factory=lambda: datetime.datetime.now()
  )

  @staticmethod
  def Success(id: str, data: Any, message: Optional[str]):
    return TaskResponse(
      data=data,
      message=message,
      status=TaskStatus.Success,
      id=id,
    )

  @staticmethod
  def Pending(id: str, message: str):
    return TaskResponse(
      data=TaskResponseData.Empty(),
      message=message,
      status=TaskStatus.Pending,
      id=id,
    )

  @staticmethod
  def Error(id: str, error_message: str):
    return TaskResponse(
      data=TaskResponseData.Empty(),
      message=error_message,
      status=TaskStatus.Failed,
      id=id,
    )
  
  @staticmethod
  def Idle(id: str):
    return TaskResponse(
      data=TaskResponseData.Empty(),
      message=None,
      status=TaskStatus.Idle,
      id=id,
    )