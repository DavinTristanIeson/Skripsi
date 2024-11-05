from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Literal, Union
import pydantic

class IPCRequestType(str, Enum):
  TopicModeling = "topic_modeling"
  Topics = "topics"
  MergeTopics = "merge_topics"
  DeleteTopics = "delete_topic"
  CreateTopic = "create_topic"
  TopicSimilarity = "topic_similarity"
  Association = "association"
  Evaluation = "evaluation"

class IPCRequestBase(pydantic.BaseModel):
  project_id: str
  id: str

class IPCRequestData(SimpleNamespace):
  class TopicModeling(IPCRequestBase):
    type: Literal[IPCRequestType.TopicModeling] = IPCRequestType.TopicModeling

    @staticmethod
    def task_id(project_id: str):
      return f"topic-modeling: {project_id}"

  class Topics(IPCRequestBase):
    type: Literal[IPCRequestType.Topics] = IPCRequestType.Topics
    column: str
    @staticmethod
    def task_id(project_id: str, column: str):
      return f"{project_id}_{column}_topics"

  class TopicSimilarityPlot(IPCRequestBase):
    type: Literal[IPCRequestType.TopicSimilarity] = IPCRequestType.TopicSimilarity
    column: str
    @staticmethod
    def task_id(project_id: str, column: str):
      return f"{project_id}_{column}_topic-similarity"

  class MergeTopics(IPCRequestBase):
    type: Literal[IPCRequestType.MergeTopics] = IPCRequestType.MergeTopics
    topics: list[int]
    @staticmethod
    def task_id(project_id: str):
      return f"{project_id}_merge-topics"

  class CreateTopic(IPCRequestBase):
    type: Literal[IPCRequestType.CreateTopic] = IPCRequestType.CreateTopic
    documents: list[int]
    @staticmethod
    def task_id(project_id: str):
      return f"{project_id}_create-topic"

  class DeleteTopics(IPCRequestBase):
    type: Literal[IPCRequestType.DeleteTopics] = IPCRequestType.DeleteTopics
    topics: list[int]
    @staticmethod
    def task_id(project_id: str):
      return f"{project_id}_delete-topics"

  class Association(IPCRequestBase):
    type: Literal[IPCRequestType.Association] = IPCRequestType.Association
    column1: str
    column2: str
    @staticmethod
    def task_id(project_id: str, column1: str, column2: str):
      return f"{project_id}_{column1}_{column2}_association"
  
  class Evaluation(IPCRequestBase):
    type: Literal[IPCRequestType.Evaluation] = IPCRequestType.Evaluation
    column: str
    @staticmethod
    def task_id(project_id: str, column: str):
      return f"{project_id}_{column}_evaluation"
  

IPCRequest = Union[
  IPCRequestData.TopicModeling,
  IPCRequestData.Topics,
  IPCRequestData.TopicSimilarityPlot,
  IPCRequestData.MergeTopics,
  IPCRequestData.CreateTopic,
  IPCRequestData.DeleteTopics,
  IPCRequestData.Association,
  IPCRequestData.Evaluation,
]

class IPCRequestWrapper(pydantic.RootModel):
  root: Annotated[
    IPCRequest,
    pydantic.Field(discriminator="type")
  ]
  
__all__ = [
  "IPCRequestType",
  "IPCRequestData",
  "IPCRequest",
  "IPCRequestWrapper",
]