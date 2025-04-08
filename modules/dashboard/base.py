from enum import Enum
from typing import Any
import pydantic

class DashboardItemRect(pydantic.BaseModel):
  x: int = pydantic.Field(ge=0)
  y: int = pydantic.Field(ge=0)
  width: int = pydantic.Field(gt=0)
  height: int = pydantic.Field(gt=0)

class BaseDashboardItem(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)
  column: str
  rect: DashboardItemRect
  config: Any

class DashboardTypeEnum(str, Enum):
  Histogram = "histogram"
  LinePlot = "line-plot"
  DescriptiveStatistics = "descriptive-statistics"
  BoxPlot = "box-plot"

__all__ = [
  "BaseDashboardItem",
  "DashboardTypeEnum"
]