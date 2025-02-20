from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Literal, Union
import pydantic

class TaskRequestType(str, Enum):
  TopicModeling = "topic_modeling"


class TaskRequestData(SimpleNamespace):
  class TopicModeling(pydantic.BaseModel):
    type: Literal[TaskRequestType.TopicModeling] = TaskRequestType.TopicModeling

    @staticmethod
    def task_id(project_id: str):
      return f"topic-modeling: {project_id}"
    
  TypeUnion = Union[
    TopicModeling,
  ]
  DiscriminatedUnion = Annotated[TypeUnion, pydantic.Field(discriminator="type")]

class TaskRequest(pydantic.BaseModel):
  project_id: str
  id: str
  data: TaskRequestData.DiscriminatedUnion
  
__all__ = [
  "TaskRequestType",
  "TaskRequestData",
  "TaskRequest",
]