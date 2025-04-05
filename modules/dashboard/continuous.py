from typing import Literal, Optional, Union
from modules.dashboard.base import BaseDashboardItem, DashboardTypeEnum
from modules.table.filter import TableSort

class HistogramDashboardItem(BaseDashboardItem):
  type: Literal[DashboardTypeEnum.Histogram]
  bins: Optional[int]

class LinePlotDashboardItem(BaseDashboardItem):
  type: Literal[DashboardTypeEnum.LinePlot]
  sort: Optional[TableSort]

class DescriptiveStatisticsDashboardItem(BaseDashboardItem):
  type: Literal[DashboardTypeEnum.DescriptiveStatistics]

class BoxPlotDashboardItem(BaseDashboardItem):
  type: Literal[DashboardTypeEnum.BoxPlot]

ContinuousDashboardItemUnion = Union[
  HistogramDashboardItem,
  LinePlotDashboardItem,
  DescriptiveStatisticsDashboardItem,
  BoxPlotDashboardItem
]

__all__ = [
  "HistogramDashboardItem",
  "LinePlotDashboardItem",
  "DescriptiveStatisticsDashboardItem",
  "BoxPlotDashboardItem",
  "ContinuousDashboardItemUnion"
]