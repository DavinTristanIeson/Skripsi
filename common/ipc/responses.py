import datetime
from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Any, Literal, Optional, Sequence, Union

import pandas as pd
import pydantic
import numpy.typing as npt

# ENUMS
class IPCResponseDataType(str, Enum):
  Plot = "plot"
  Topics = "topics"
  Association = "association"
  Empty = "empty"

class AssociationDataType(str, Enum):
  Categorical = "categorical"
  Continuous = "continuous"
  Temporal = "temporal"

class IPCResponseStatus(str, Enum):
  Idle = "idle"
  Pending = "pending"
  Success = "success"
  Failed = "failed"

# OTHER DATA
class AssociationData(SimpleNamespace):
  class Categorical(pydantic.BaseModel):
    type: Literal[AssociationDataType.Categorical] = AssociationDataType.Categorical
    crosstab_heatmap: str
    association_heatmap: str
    biplot: str

    topics: Sequence[str]
    # Column 2 outcomes
    outcomes: Sequence[str]

    # CSV
    crosstab_csv: str
    association_csv: str

  class Continuous(pydantic.BaseModel):
    type: Literal[AssociationDataType.Continuous] = AssociationDataType.Continuous
    plot: str
    topics: Sequence[str]

    # CSV
    statistics_csv: str

  class Temporal(pydantic.BaseModel):
    type: Literal[AssociationDataType.Temporal] = AssociationDataType.Temporal
    plot: str
    topics: Sequence[str]
    bins: Sequence[str]

  TypeUnion = Union[Categorical, Continuous, Temporal]
  DiscriminatedUnion = Annotated[TypeUnion, pydantic.Field(discriminator="type")]

# IPC RESPONSE

class IPCProgressReport(pydantic.BaseModel):
  progress: float
  message: Optional[str]
  timestamp: datetime.datetime = pydantic.Field(
    default_factory=lambda: datetime.datetime.now()
  )
  
  @pydantic.field_serializer("timestamp", when_used="json")
  def serialize__time(self, timestamp: datetime.datetime):
    return timestamp.timestamp()
  
  @pydantic.field_validator("timestamp", mode="before")
  def validate__time(cls, timestamp: int):
    if timestamp is None or not isinstance(timestamp, int):
      return datetime.datetime.now()
    return datetime.datetime.fromtimestamp(timestamp)



class IPCResponseData(SimpleNamespace):
  class Plot(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Plot] = IPCResponseDataType.Plot
    plot: str

  class Topics(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Topics] = IPCResponseDataType.Topics
    plot: str
    topic_words: dict[str, Sequence[tuple[str, float]]]

  class Association(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Association] = IPCResponseDataType.Association
    data: AssociationData.DiscriminatedUnion

  class Empty(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Empty] = IPCResponseDataType.Empty

  TypeUnion = Union[
    Plot,
    Empty,
    Topics,
    Association,
  ]
  DiscriminatedUnion = Annotated[TypeUnion, pydantic.Field(discriminator="type")]

class IPCResponse(pydantic.BaseModel):
  id: str
  data: IPCResponseData.DiscriminatedUnion
  status: IPCResponseStatus
  message: Optional[str] = None
  progress: Optional[float] = None
  error: Optional[str] = None

  @staticmethod
  def Success(id: str, data: Any, message: Optional[str]):
    return IPCResponse(
      data=data,
      error=None,
      message=message,
      progress=1,
      status=IPCResponseStatus.Success,
      id=id,
    )

  @staticmethod
  def Pending(id: str, progress: float, message: str):
    return IPCResponse(
      data=IPCResponseData.Empty(),
      error=None,
      message=message,
      progress=progress,
      status=IPCResponseStatus.Pending,
      id=id,
    )

  @staticmethod
  def Error(id: str, error_message: str):
    return IPCResponse(
      data=IPCResponseData.Empty(),
      error=error_message,
      message=None,
      progress=1,
      status=IPCResponseStatus.Failed,
      id=id,
    )


__all__ = [
  "IPCResponseData",
  "IPCResponseStatus",
  "IPCResponse"
]