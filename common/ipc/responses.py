import datetime
from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Any, Literal, Optional, Sequence, Union

import pydantic

from common.models.enum import ExposedEnum
from wordsmith.topic.evaluation import ColumnTopicsEvaluationResult


# ENUMS
class IPCResponseDataType(str, Enum):
  Plot = "plot"
  Topics = "topics"
  TopicSimilarity = "topic_similarity"
  Association = "association"
  Empty = "empty"
  Evaluation = "evaluation"

class AssociationDataTypeEnum(str, Enum):
  Categorical = "categorical"
  Continuous = "continuous"
  Temporal = "temporal"

ExposedEnum().register(AssociationDataTypeEnum)

class IPCResponseStatus(str, Enum):
  Idle = "idle"
  Pending = "pending"
  Success = "success"
  Failed = "failed"

# OTHER DATA
class AssociationData(SimpleNamespace):
  class Categorical(pydantic.BaseModel):
    type: Literal[AssociationDataTypeEnum.Categorical] = AssociationDataTypeEnum.Categorical
    crosstab_heatmap: str = pydantic.Field(repr=False)
    residual_heatmap: str = pydantic.Field(repr=False)
    biplot: str = pydantic.Field(repr=False)

    topics: Sequence[str]
    # Column 2 outcomes
    outcomes: Sequence[str]

    # CSV
    crosstab_csv: str = pydantic.Field(repr=False)
    association_csv: str = pydantic.Field(repr=False)

  class Continuous(pydantic.BaseModel):
    type: Literal[AssociationDataTypeEnum.Continuous] = AssociationDataTypeEnum.Continuous
    violin_plot: str = pydantic.Field(repr=False)
    topics: Sequence[str]

    # CSV
    statistics_csv: str

  class Temporal(pydantic.BaseModel):
    type: Literal[AssociationDataTypeEnum.Temporal] = AssociationDataTypeEnum.Temporal
    line_plot: str
    topics: Sequence[str]
    
  TypeUnion = Union[Categorical, Continuous, Temporal]
  DiscriminatedUnion = Annotated[TypeUnion, pydantic.Field(discriminator="type")]

# IPC RESPONSE

class IPCResponseData(SimpleNamespace):
  class Plot(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Plot] = IPCResponseDataType.Plot
    plot: str

  class TopicSimilarity(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.TopicSimilarity] = IPCResponseDataType.TopicSimilarity
    column: str
    topics: Sequence[str]
    heatmap: str = pydantic.Field(repr=False)
    ldavis: str = pydantic.Field(repr=False)
    similarity_matrix: Sequence[Sequence[float]] = pydantic.Field(repr=False)
    dendrogram: str = pydantic.Field(repr=False)
    scatterplot: str = pydantic.Field(repr=False)

  class Topics(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Topics] = IPCResponseDataType.Topics
    column: str
    topics: Sequence[str]
    topic_words: Sequence[Sequence[tuple[str, float]]]
    frequencies: Sequence[int]
    total: int
    outliers: int
    invalid: int
    valid: int
    topics_barchart: str = pydantic.Field(repr=False)
    frequency_barchart: str = pydantic.Field(repr=False)

  class Association(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Association] = IPCResponseDataType.Association
    column1: str
    column2: str
    excluded: int
    excluded_left: int
    excluded_right: int
    total: int
    data: AssociationData.DiscriminatedUnion

  class Empty(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Empty] = IPCResponseDataType.Empty

  class Evaluation(ColumnTopicsEvaluationResult, pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Evaluation] = IPCResponseDataType.Evaluation

  TypeUnion = Union[
    Plot,
    Empty,
    Topics,
    Association,
    TopicSimilarity,
    Evaluation
  ]
  DiscriminatedUnion = Annotated[TypeUnion, pydantic.Field(discriminator="type")]

class IPCResponse(pydantic.BaseModel):
  id: str
  data: IPCResponseData.DiscriminatedUnion
  status: IPCResponseStatus
  message: Optional[str] = None
  progress: float = 0
  timestamp: datetime.datetime = pydantic.Field(
    default_factory=lambda: datetime.datetime.now()
  )

  @staticmethod
  def Success(id: str, data: Any, message: Optional[str]):
    return IPCResponse(
      data=data,
      message=message,
      progress=1,
      status=IPCResponseStatus.Success,
      id=id,
    )

  @staticmethod
  def Pending(id: str, progress: float, message: str):
    return IPCResponse(
      data=IPCResponseData.Empty(),
      message=message,
      progress=progress,
      status=IPCResponseStatus.Pending,
      id=id,
    )

  @staticmethod
  def Error(id: str, error_message: str):
    return IPCResponse(
      data=IPCResponseData.Empty(),
      message=error_message,
      progress=1,
      status=IPCResponseStatus.Failed,
      id=id,
    )
  
  @staticmethod
  def Idle(id: str):
    return IPCResponse(
      data=IPCResponseData.Empty(),
      message=None,
      progress=0,
      status=IPCResponseStatus.Idle,
      id=id,
    )


__all__ = [
  "IPCResponseData",
  "IPCResponseStatus",
  "IPCResponse"
]