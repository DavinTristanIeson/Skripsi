import abc
from enum import Enum
from typing import Optional, Sequence

import pandas as pd
import pydantic

from modules.api.enum import ExposedEnum


class SchemaColumnTypeEnum(str, Enum):
  Continuous = "continuous"
  Categorical = "categorical"
  OrderedCategorical = "ordered-categorical"
  Temporal = "temporal"
  Textual = "textual"
  Unique = "unique"
  Geospatial = "geospatial"
  Boolean = "boolean"

  # Internal
  Topic = "topic"

CATEGORICAL_SCHEMA_COLUMN_TYPES = [SchemaColumnTypeEnum.Categorical, SchemaColumnTypeEnum.OrderedCategorical, SchemaColumnTypeEnum.Temporal, SchemaColumnTypeEnum.Topic, SchemaColumnTypeEnum.Boolean]
ORDERED_SCHEMA_COLUMN_TYPES = [SchemaColumnTypeEnum.Temporal, SchemaColumnTypeEnum.Continuous, SchemaColumnTypeEnum.OrderedCategorical]
ORDERED_CATEGORICAL_SCHEMA_COLUMN_TYPES = [SchemaColumnTypeEnum.Temporal, SchemaColumnTypeEnum.OrderedCategorical]
ANALYZABLE_SCHEMA_COLUMN_TYPES = [SchemaColumnTypeEnum.Categorical, SchemaColumnTypeEnum.Continuous, SchemaColumnTypeEnum.OrderedCategorical, SchemaColumnTypeEnum.Temporal, SchemaColumnTypeEnum.Topic, SchemaColumnTypeEnum.Boolean]

ExposedEnum().register(SchemaColumnTypeEnum)

class GeospatialRoleEnum(str, Enum):
  Latitude = "latitude"
  Longitude = "longitude"

ExposedEnum().register(GeospatialRoleEnum)

# Frozen classes are required for easy columnar comparison in schema_manager.py.

class _BaseSchemaColumn(pydantic.BaseModel, abc.ABC, frozen=True):
  model_config = pydantic.ConfigDict(use_enum_values=True)
  name: str
  description: Optional[str] = None
  internal: bool = pydantic.Field(default=False)
  source_name: Optional[str] = pydantic.Field(default=None)

  def get_internal_columns(self)->Sequence["_BaseSchemaColumn"]:
    return []

  @property
  def is_ordered(self)->bool:
    return False
  
  @property
  def is_categorical(self)->bool:
    return False

  @abc.abstractmethod
  def fit(self, df: pd.DataFrame)->None:
    ...

__all__ = [
  "GeospatialRoleEnum",
  "SchemaColumnTypeEnum"
]