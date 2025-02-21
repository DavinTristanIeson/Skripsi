import datetime
from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Literal, Union
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

class TaskResponseData(SimpleNamespace):
  class Empty(pydantic.BaseModel):
    type: Literal[TaskResponseType.Empty] = TaskResponseType.Empty

  TypeUnion = Union[Empty]
  DiscriminatedUnion = Annotated[TypeUnion, pydantic.Field(discriminator="type")]

class TaskLog(pydantic.BaseModel):
  status: TaskStatusEnum
  message: str
  timestamp: datetime.datetime = pydantic.Field(
    default_factory=lambda: datetime.datetime.now()
  )
class TaskResponse(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(use_enum_values=True)
  
  id: str
  data: TaskResponseData.DiscriminatedUnion
  logs: list[TaskLog]
  status: TaskStatusEnum

  @staticmethod
  def Idle(id: str):
    return TaskResponse(
      data=TaskResponseData.Empty(),
      logs=[],
      status=TaskStatusEnum.Idle,
      id=id,
    )
  
__all__ = [
  "TaskLog",
  "TaskResponse",
  "TaskResponseData",
  "TaskStatusEnum",
  "TaskResponseType"
]