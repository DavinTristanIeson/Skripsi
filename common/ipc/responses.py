from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Literal, Optional, Sequence, Union

import pydantic
import numpy.typing as npt

class IPCResponseDataType(str, Enum):
  Plot = "plot"
  Empty = "empty"

class IPCResponseData(SimpleNamespace):
  class Plot(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Plot] = IPCResponseDataType.Plot
    plot: str

  class TopicPlot(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Plot] = IPCResponseDataType.Plot
    plot: str
    topic_words: dict[str, Sequence[tuple[str, float]]]

  class CategoricalAssociationPlot(pydantic.BaseModel):
    crosstab_heatmap: str
    association_heatmap: str
    biplot: str
    yaxis: Sequence[str]
    xaxis: Sequence[str]
    crosstab: npt.NDArray
    association: npt.NDArray

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