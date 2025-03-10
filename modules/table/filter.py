import abc
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import pydantic

from modules.api import ExposedEnum
from modules.config import Config, SchemaColumnTypeEnum

class TableFilterTypeEnum(str, Enum):
  # Special
  And = "and"
  Or = "or"
  Not = "not"
  
  # All
  Empty = "empty"
  NotEmpty = "not_empty"
  EqualTo = "equal_to"
  IsOneOf = "is_one_of"

  # Ordered
  GreaterThan = "greater_than"
  LessThan = "less_than"
  GreaterThanOrEqualTo = "greater_than_or_equal_to"
  LessThanOrEqualTo = "less_than_or_equal_to"

  # Textual
  HasText = "has_text"

  # Multi-Categorical
  Includes = "includes"
  Excludes = "excludes"
  Only = "only"

ExposedEnum().register(TableFilterTypeEnum)

__ALLOWED_FILTER_TYPES_FOR_ALL_COLUMNS = [
  TableFilterTypeEnum.Empty,
  TableFilterTypeEnum.NotEmpty,
  TableFilterTypeEnum.EqualTo,
  TableFilterTypeEnum.IsOneOf,
]
__ALLOWED_FILTER_TYPES_FOR_ORDERED_COLUMNS = [
  *__ALLOWED_FILTER_TYPES_FOR_ALL_COLUMNS,
  TableFilterTypeEnum.GreaterThan,
  TableFilterTypeEnum.GreaterThanOrEqualTo,
  TableFilterTypeEnum.LessThan,
  TableFilterTypeEnum.LessThanOrEqualTo,
]
__ALLOWED_FILTER_TYPES_FOR_TEXTUAL_COLUMNS = [
  *__ALLOWED_FILTER_TYPES_FOR_ALL_COLUMNS,
  TableFilterTypeEnum.HasText,
]
ALLOWED_FILTER_TYPES_FOR_COLUMNS = {
  SchemaColumnTypeEnum.Categorical: __ALLOWED_FILTER_TYPES_FOR_TEXTUAL_COLUMNS,
  SchemaColumnTypeEnum.OrderedCategorical: [
    *__ALLOWED_FILTER_TYPES_FOR_TEXTUAL_COLUMNS,
    *__ALLOWED_FILTER_TYPES_FOR_ORDERED_COLUMNS,
  ],
  SchemaColumnTypeEnum.Textual: __ALLOWED_FILTER_TYPES_FOR_TEXTUAL_COLUMNS,
  SchemaColumnTypeEnum.Unique: __ALLOWED_FILTER_TYPES_FOR_TEXTUAL_COLUMNS,
  SchemaColumnTypeEnum.Continuous: __ALLOWED_FILTER_TYPES_FOR_ORDERED_COLUMNS,
  SchemaColumnTypeEnum.Geospatial: __ALLOWED_FILTER_TYPES_FOR_ORDERED_COLUMNS,
  SchemaColumnTypeEnum.Temporal: __ALLOWED_FILTER_TYPES_FOR_ORDERED_COLUMNS,
  SchemaColumnTypeEnum.MultiCategorical: [
    *__ALLOWED_FILTER_TYPES_FOR_ALL_COLUMNS,
    TableFilterTypeEnum.Excludes,
    TableFilterTypeEnum.Includes,
    TableFilterTypeEnum.Only,
  ],
}

@dataclass
class _TableFilterParams:
  data: pd.DataFrame
  config: Config
  
class _BaseTableFilter(pydantic.BaseModel, abc.ABC, frozen=True):
  target: str
  type: Any
  model_config = pydantic.ConfigDict(use_enum_values=True)
 
  @abc.abstractmethod
  def apply(self, params: _TableFilterParams)->pd.Series | np.ndarray:
    pass

class _BaseCompoundTableFilter(pydantic.BaseModel, abc.ABC, frozen=True):
  @abc.abstractmethod
  def apply(self, params: _TableFilterParams)->pd.Series | np.ndarray:
    pass

class TableSort(pydantic.BaseModel, abc.ABC, frozen=True):
  name: str
  asc: bool


__all__ = [
  "TableSort",
  "TableFilterTypeEnum",
  "ALLOWED_FILTER_TYPES_FOR_COLUMNS"
]