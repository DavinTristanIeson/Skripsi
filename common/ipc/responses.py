from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Literal, Optional, Union

import pydantic
import pandas as pd

class IPCResponseDataType(str, Enum):
  Plot = "plot"
  Empty = "empty"

class IPCResponseData(SimpleNamespace):
  class Plot(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Plot] = IPCResponseDataType.Plot
    plot: str
  class Empty(pydantic.BaseModel):
    type: Literal[IPCResponseDataType.Empty] = IPCResponseDataType.Empty

class IPCResponseStatus(str, Enum):
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
  error: Optional[str] = None


__all__ = [
  "IPCResponseData",
  "IPCResponseStatus",
  "IPCResponse"
]