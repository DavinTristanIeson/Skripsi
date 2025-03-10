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
  MultiCategorical = "multi-categorical"
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

class _BaseSchemaColumn(pydantic.BaseModel, abc.ABC):
  model_config = pydantic.ConfigDict(use_enum_values=True, frozen=True)
  name: str
  alias: Optional[str] = None
  internal: bool = pydantic.Field(default=False, exclude=True)

  def get_internal_columns(self)->Sequence["_BaseSchemaColumn"]:
    return []

  @abc.abstractmethod
  def fit(self, df: pd.DataFrame)->None:
    ...


class _BaseMultiCategoricalSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel, abc.ABC):
  delimiter: str = ","
  is_json: bool = True
    
  def json2list(self, data: Sequence[str]):
    import orjson
    for row in data:
      row_categories: list[str]
      if self.is_json:
        row_categories = list(orjson.loads(row))
      else:
        row_categories = list(map(
          lambda category: category.strip(),
          str(row).split(self.delimiter)
        ))
      yield row_categories

  def count_categories(self, data: Iterable[Sequence[str]])->Counter[str]:
    global_counter = Counter()
    for row_categories in data:
      global_counter += Counter(row_categories)
    return global_counter
  
  def flatten(self, data: Iterable[Sequence[str]]):
    for row_categories in data:
      for category in row_categories:
        yield category
    
  def list2json(self, tags_list: Iterable[Sequence[str]]):
    import orjson
    return list(map(
      lambda tags: orjson.dumps(tags),
      tags_list
    ))


__all__ = [
  "GeospatialRoleEnum",
  "SchemaColumnTypeEnum"
]