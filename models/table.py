from typing import Any, Optional
import pydantic
from modules.comparison.effect_size import EffectSizeMethodEnum
from modules.comparison.statistic_test import StatisticTestMethodEnum
from modules.config import SchemaColumn
from modules.table import NamedTableFilter, TableFilter

# Schema
class GetTableColumnSchema(pydantic.BaseModel):
  column: str
  filter: Optional[TableFilter]

class GetTableGeographicalColumnSchema(pydantic.BaseModel):
  filter: Optional[TableFilter]
  latitude: str
  longitude: str

class TableColumnsStatisticTestSchema(pydantic.BaseModel):
  group1: NamedTableFilter
  group2: NamedTableFilter
  column: str

  statistic_test_preference: StatisticTestMethodEnum
  effect_size_preference: EffectSizeMethodEnum
  exclude_overlapping_rows: bool

# Resources

class TableColumnValuesResource(pydantic.BaseModel):
  column: SchemaColumn
  values: list[Any]

class TableColumnFrequencyDistributionResource(pydantic.BaseModel):
  column: SchemaColumn
  values: list[str]
  frequencies: list[int]

class TableColumnGeographicalPointsResource(pydantic.BaseModel):
  latitude_column: SchemaColumn
  longitude_column: SchemaColumn
  latitude: list[float]
  longitude: list[float]
  sizes: list[int]

class TableColumnCountsResource(pydantic.BaseModel):
  column: SchemaColumn
  total: int
  valid: int
  invalid: int
  # Only for topics
  outlier: Optional[int]