from typing import Any, Optional
import pydantic
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import SchemaColumn
from modules.table.filter_variants import TableFilter

# Schema
class GetTableColumnSchema(pydantic.BaseModel):
  column: str
  filter: Optional[TableFilter]

class GetTableGeographicalColumnSchema(pydantic.BaseModel):
  filter: Optional[TableFilter]
  latitude: str
  longitude: str

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