from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Literal, Union
import pydantic

from common.models.enum import EnumMemberDescriptor, ExposedEnum

class TopicSimilarityVisualizationMethodEnum(str, Enum):
  Heatmap = "heatmap"
  Ldavis = "ldavis"

ExposedEnum().register(TopicSimilarityVisualizationMethodEnum, {
  TopicSimilarityVisualizationMethodEnum.Heatmap: EnumMemberDescriptor(
    label="Heatmap",
    description="The relationship between each topic is represented as a matrix; where the i-th row and the j-th column represents the relatedness of Topic i and Topic j. Brighter cell color means both topics are closely related.",
  ),
  TopicSimilarityVisualizationMethodEnum.Ldavis: EnumMemberDescriptor(
    label="LDAVis-style",
    description="Topics are represented as bubbles in a plot, where the size of the bubbles represent the number of documents assigned to that topic. Furthermore, bubbles that are closely related are placed near each other.",
  ),
})

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
    visualization: TopicSimilarityVisualizationMethodEnum


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