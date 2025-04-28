from enum import Enum
from typing import Any, Optional
import numpy as np
import pandas as pd
import pydantic
from modules.api.enum import ExposedEnum
from modules.config import SchemaColumn
from modules.table import TableFilter
from modules.topic.model import Topic

# Schema
class GetTableColumnSchema(pydantic.BaseModel):
  column: str
  filter: Optional[TableFilter]

class TableColumnAggregateMethodEnum(str, Enum):
  Sum = "sum"
  Mean = "mean"
  Median = "median"
  StandardDeviation = "std-dev"
  Max = "max"
  Min = "min"

ExposedEnum().register(TableColumnAggregateMethodEnum)

class GetTableColumnAggregateValuesSchema(GetTableColumnSchema, pydantic.BaseModel):
  grouped_by: str
  method: TableColumnAggregateMethodEnum

class DatasetFilterSchema(pydantic.BaseModel):
  filter: Optional[TableFilter]

class GetTableGeographicalColumnSchema(pydantic.BaseModel):
  filter: Optional[TableFilter]
  latitude_column: str
  longitude_column: str
  label_column: Optional[str]


class GetTableGeographicalAggregateValuesSchema(pydantic.BaseModel):
  filter: Optional[TableFilter]
  latitude_column: str
  longitude_column: str
  target_column: str
  label_column: Optional[str]

  method: TableColumnAggregateMethodEnum


# Resources

class TableColumnValuesResource(pydantic.BaseModel):
  column: SchemaColumn
  values: list[Any]

class TableColumnFrequencyDistributionResource(pydantic.BaseModel):
  column: SchemaColumn
  categories: list[str]
  frequencies: list[int]

class TableColumnAggregateValuesResource(pydantic.BaseModel):
  column: SchemaColumn
  categories: list[str]
  values: list[float]

class TableColumnGeographicalPointsResource(pydantic.BaseModel):
  latitude_column: SchemaColumn
  longitude_column: SchemaColumn
  latitudes: list[float]
  longitudes: list[float]
  labels: Optional[list[str]]
  values: list[int | float]

class TableColumnCountsResource(pydantic.BaseModel):
  column: SchemaColumn
  total: int
  inside: int
  outside: int
  valid: int
  invalid: int
  # Only for topics
  outlier: Optional[int]

class TableWordItemResource(pydantic.BaseModel):
  group: int
  word: str
  size: int

class TableWordsResource(pydantic.BaseModel):
  column: SchemaColumn
  words: list[TableWordItemResource]

class TableTopicsResource(pydantic.BaseModel):
  column: SchemaColumn
  topics: list[Topic]

class DescriptiveStatisticsResource(pydantic.BaseModel):
  count: float
  mean: float
  median: float
  std: float
  min: float
  q1: float
  q3: float
  max: float
  inlier_range: tuple[float, float]
  outlier_count: int

  @staticmethod
  def from_series(column: pd.Series):
    summary = column.describe()
    iqr = summary["75%"] - summary["25%"]
    inlier_range = (summary["25%"] - (1.5 * iqr), summary["75%"] + (1.5 * iqr))

    inlier_mask = np.bitwise_or(column >= inlier_range[0], column <= inlier_range[1])
    outlier_count = column[~inlier_mask].count()
    return DescriptiveStatisticsResource(
      count=summary["count"],
      mean=summary["mean"],
      median=summary["50%"],
      std=summary["std"],
      min=summary["min"],
      q1=summary["25%"],
      q3=summary["75%"],
      max=summary["max"],
      inlier_range=inlier_range,
      outlier_count=outlier_count,
    )
  
class TableDescriptiveStatisticsResource(pydantic.BaseModel):
  column: SchemaColumn
  statistics: DescriptiveStatisticsResource