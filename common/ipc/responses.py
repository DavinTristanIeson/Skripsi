import datetime
from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Any, Literal, Optional, Sequence, Union

import pydantic

from common.models.enum import EnumMemberDescriptor, ExposedEnum


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

ExposedEnum().register(AssociationDataTypeEnum, {
  AssociationDataTypeEnum.Categorical: EnumMemberDescriptor(
    label="Categorical",
    description="Discrete variables that only consist of a few unique outcomes or values."
  ),
  AssociationDataTypeEnum.Continuous: EnumMemberDescriptor(
    label="Continuous",
    description="Variables that do not have a defined level of precision."
  ),
  AssociationDataTypeEnum.Temporal: EnumMemberDescriptor(
    label="Temporal",
    description="Variables that concern time."
  )
})

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
    bins: Sequence[str]

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

  class Topics(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Topics] = IPCResponseDataType.Topics
    column: str
    plot: str = pydantic.Field(repr=False)
    topics: Sequence[str]
    topic_words: Sequence[Sequence[tuple[str, float]]]
    frequencies: Sequence[int]
    total: int
    outliers: int

  class Association(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Association] = IPCResponseDataType.Association
    column1: str
    column2: str
    data: AssociationData.DiscriminatedUnion

  class Empty(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Empty] = IPCResponseDataType.Empty

  class Evaluation(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Evaluation] = IPCResponseDataType.Evaluation
    column: str
    topics: Sequence[str]
    cv_score: float
    topic_diversity_score: float
    cv_topic_scores: Sequence[float]
    cv_barchart: str

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