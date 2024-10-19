from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Literal, Optional, Sequence, Union

import pandas as pd
import pydantic
import numpy.typing as npt

class IPCResponseDataType(str, Enum):
  Plot = "plot"
  TopicPlot = "topic_plot"
  CategoricalAssociationPlot = "categorical_association_plot"
  ContinuousAssociationPlot = "continuous_association_plot"
  TemporalAssociationPlot = "continuous_association_plot"
  Empty = "empty"

class IPCResponseData(SimpleNamespace):
  class Plot(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Plot] = IPCResponseDataType.Plot
    plot: str

  class TopicPlot(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.TopicPlot] = IPCResponseDataType.TopicPlot
    plot: str
    topic_words: dict[str, Sequence[tuple[str, float]]]

  class CategoricalAssociationPlot(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.CategoricalAssociationPlot] = IPCResponseDataType.CategoricalAssociationPlot
    crosstab_heatmap: str
    association_heatmap: str
    biplot: str

    topics: Sequence[str]
    # Column 2 outcomes
    outcomes: Sequence[str]
    crosstab: npt.NDArray
    association: npt.NDArray

  class ContinuousAssociationPlot(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.ContinuousAssociationPlot] = IPCResponseDataType.ContinuousAssociationPlot
    plot: str
    topics: Sequence[str]
    statistics: pd.DataFrame

  class TemporalAssociationPlot(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.TemporalAssociationPlot] = IPCResponseDataType.TemporalAssociationPlot
    plot: str
    topics: Sequence[str]
    bins: Sequence[str]


  class Empty(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Empty] = IPCResponseDataType.Empty

class IPCResponseStatus(str, Enum):
  Idle = "idle"
  Pending = "pending"
  Success = "success"
  Failed = "failed"

IPCResponseDataUnion = Union[
  IPCResponseData.Plot,
  IPCResponseData.Empty,
]

class IPCResponse(pydantic.BaseModel):
  id: str
  data: Annotated[IPCResponseDataUnion, pydantic.Field(discriminator="type")]
  status: IPCResponseStatus
  message: Optional[str] = None
  progress: Optional[float] = None
  error: Optional[str] = None


__all__ = [
  "IPCResponseData",
  "IPCResponseStatus",
  "IPCResponse"
]