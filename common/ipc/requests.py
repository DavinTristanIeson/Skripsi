from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Literal, Union
import uuid
import pydantic

class TopicSimilarityVisualizationMethod(str, Enum):
  Heatmap = "heatmap"
  Ldavis = "ldavis"

class IPCRequestType(str, Enum):
  TopicModeling = "topic_modeling"
  CancelTask = "cancel_task"
  TopicPlot = "topic_sunburst_plot"
  MergeTopics = "merge_topics"
  DeleteTopics = "delete_topic"
  CreateTopic = "create_topic"
  TopicCorrelationPlot = "topic_correlation_plot"
  AssociationPlot = "association_plot"

class IPCRequestBase(pydantic.BaseModel):
  project_id: str
  id: str

class IPCRequestData(SimpleNamespace):
  class CancelTask(pydantic.BaseModel):
    type: Literal[IPCRequestType.CancelTask] = IPCRequestType.CancelTask
    id: str
  class TopicModeling(IPCRequestBase):
    type: Literal[IPCRequestType.TopicModeling] = IPCRequestType.TopicModeling

  class TopicPlot(IPCRequestBase):
    type: Literal[IPCRequestType.TopicPlot] = IPCRequestType.TopicPlot
    col: str
    topic: int

  class TopicCorrelationPlot(IPCRequestBase):
    type: Literal[IPCRequestType.TopicCorrelationPlot] = IPCRequestType.TopicCorrelationPlot
    col: str
    visualization: TopicSimilarityVisualizationMethod


  class MergeTopics(IPCRequestBase):
    type: Literal[IPCRequestType.MergeTopics] = IPCRequestType.MergeTopics
    topics: list[int]

  class CreateTopic(IPCRequestBase):
    type: Literal[IPCRequestType.CreateTopic] = IPCRequestType.CreateTopic
    documents: list[int]

  class DeleteTopics(IPCRequestBase):
    type: Literal[IPCRequestType.DeleteTopics] = IPCRequestType.DeleteTopics
    topics: list[int]

  class AssociationPlot(IPCRequestBase):
    type: Literal[IPCRequestType.AssociationPlot] = IPCRequestType.AssociationPlot
    col1: str
    col2: str
  

IPCRequest = Union[
  IPCRequestData.TopicModeling,
  IPCRequestData.TopicPlot,
  IPCRequestData.TopicCorrelationPlot,
  IPCRequestData.MergeTopics,
  IPCRequestData.CreateTopic,
  IPCRequestData.DeleteTopics,
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