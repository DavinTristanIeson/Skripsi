import abc
from enum import Enum
from typing import Iterable, Optional, Sequence
from collections import Counter

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

  # Internal
  Topic = "topic"

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

  @abc.abstractmethod
  def fit(self, df: pd.DataFrame)->None:
    ...

__all__ = [
  "GeospatialRoleEnum",
  "SchemaColumnTypeEnum"
]