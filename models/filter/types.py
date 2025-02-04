import abc
from enum import Enum
from typing import Any

import numpy.typing as npt
import pandas as pd
import pydantic

from common.models.enum import ExposedEnum
from common.models.validators import CommonModelConfig
from models.config.schema import SchemaColumn, SchemaColumnTypeEnum

class DatasetFilterTypeEnum(str, Enum):
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

ExposedEnum().register(DatasetFilterTypeEnum)

ALLOWED_FILTER_TYPES_FOR_ALL_COLUMNS = [
  DatasetFilterTypeEnum.Empty,
  DatasetFilterTypeEnum.NotEmpty,
  DatasetFilterTypeEnum.EqualTo,
  DatasetFilterTypeEnum.IsOneOf,
]
ALLOWED_FILTER_TYPES_FOR_ORDERED_COLUMNS = [
  *ALLOWED_FILTER_TYPES_FOR_ALL_COLUMNS,
  DatasetFilterTypeEnum.GreaterThan,
  DatasetFilterTypeEnum.GreaterThanOrEqualTo,
  DatasetFilterTypeEnum.LessThan,
  DatasetFilterTypeEnum.LessThanOrEqualTo,
]
ALLOWED_FILTER_TYPES_FOR_TEXTUAL_COLUMNS = [
  *ALLOWED_FILTER_TYPES_FOR_ALL_COLUMNS,
  DatasetFilterTypeEnum.HasText,
]
ALLOWED_FILTER_TYPES_FOR_COLUMNS = {
  SchemaColumnTypeEnum.Categorical: ALLOWED_FILTER_TYPES_FOR_TEXTUAL_COLUMNS,
  SchemaColumnTypeEnum.Textual: ALLOWED_FILTER_TYPES_FOR_TEXTUAL_COLUMNS,
  SchemaColumnTypeEnum.Unique: ALLOWED_FILTER_TYPES_FOR_TEXTUAL_COLUMNS,
  SchemaColumnTypeEnum.Image: ALLOWED_FILTER_TYPES_FOR_TEXTUAL_COLUMNS,
  SchemaColumnTypeEnum.Continuous: ALLOWED_FILTER_TYPES_FOR_ORDERED_COLUMNS,
  SchemaColumnTypeEnum.Geospatial: ALLOWED_FILTER_TYPES_FOR_ORDERED_COLUMNS,
  SchemaColumnTypeEnum.Temporal: ALLOWED_FILTER_TYPES_FOR_ORDERED_COLUMNS,
  SchemaColumnTypeEnum.MultiCategorical: [
    *ALLOWED_FILTER_TYPES_FOR_ALL_COLUMNS,
    DatasetFilterTypeEnum.Excludes,
    DatasetFilterTypeEnum.Includes,
    DatasetFilterTypeEnum.Only,
  ],
}

class DatasetFilterParams:
  data: pd.Series
  column: SchemaColumn

class BaseDatasetFilter(pydantic.BaseModel, abc.ABC):
  model_config = CommonModelConfig
  target: str
  type: Any
 
  @abc.abstractmethod
  def apply(self, params: DatasetFilterParams)->pd.Series[bool] | npt.NDArray:
    pass



