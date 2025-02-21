import abc
from enum import Enum
from typing import Sequence

import pandas as pd
import pydantic

from modules.api.enum import ExposedEnum


class SchemaColumnTypeEnum(str, Enum):
  Continuous = "continuous"
  Categorical = "categorical"
  OrderedCategorical = "ordered-categorical"
  MultiCategorical = "multi-categorical"
  Temporal = "temporal"
  Textual = "textual"
  Unique = "unique"
  Geospatial = "geospatial"
  Image = "image"

  # Internal
  Topic = "topic"

ExposedEnum().register(SchemaColumnTypeEnum)

class GeospatialRoleEnum(str, Enum):
  Latitude = "latitude"
  Longitude = "longitude"

ExposedEnum().register(GeospatialRoleEnum)

class _BaseSchemaColumn(pydantic.BaseModel, abc.ABC):
  model_config = pydantic.ConfigDict(use_enum_values=True)
  name: str
  internal: bool = pydantic.Field(default=False, exclude=True)
  active: bool = True

  def get_internal_columns(self)->Sequence["_BaseSchemaColumn"]:
    return []

  @abc.abstractmethod
  def fit(self, df: pd.DataFrame)->None:
    ...

__all__ = [
  "GeospatialRoleEnum",
  "SchemaColumnTypeEnum"
]