import functools
import json
from typing import Any, Literal, cast

import numpy as np
import pydantic
from models.config.schema import MultiCategoricalSchemaColumn, SchemaColumnTypeEnum
from models.filter.errors import DatasetFilterError
from models.filter.types import BaseDatasetFilter, DatasetFilterParams, DatasetFilterTypeEnum
from common.models.validators import CommonModelConfig
from common.logger import RegisteredLogger
from models.filter.validators import parse_value
import numpy.typing as npt

logger = RegisteredLogger().provision("Filter Controller")

class AndDatasetFilter(BaseDatasetFilter, pydantic.BaseModel):
  type: Literal[DatasetFilterTypeEnum.And]
  operands: list[BaseDatasetFilter] = pydantic.Field(min_length=1)
  def apply(self, params):
    return functools.reduce(
      lambda acc, cur: acc & cur.apply(params),
      self.operands, np.full(len(params.data), 1, dtype=np.bool_)
    )
  
class OrDatasetFilter(BaseDatasetFilter, pydantic.BaseModel):
  type: Literal[DatasetFilterTypeEnum.Or]
  operands: list[BaseDatasetFilter] = pydantic.Field(min_length=1)
  def apply(self, params):
    return functools.reduce(
      lambda acc, cur: acc | cur.apply(params),
      self.operands, np.full(len(params.data), 1, dtype=np.bool_)
    )

class NotDatasetFilter(BaseDatasetFilter, pydantic.BaseModel):
  type: Literal[DatasetFilterTypeEnum.Not]
  operand: BaseDatasetFilter
  def apply(self, params):
    return ~self.operand.apply(params)
  

class EmptyDatasetFilter(BaseDatasetFilter, pydantic.BaseModel):
  type: Literal[DatasetFilterTypeEnum.Empty]
  def apply(self, params):
    return params.data.isna()

class NotEmptyDatasetFilter(BaseDatasetFilter, pydantic.BaseModel):
  type: Literal[DatasetFilterTypeEnum.NotEmpty]
  def apply(self, params):
    return params.data.notna()

class EqualToDatasetFilter(BaseDatasetFilter, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[DatasetFilterTypeEnum.EqualTo]
  value: Any
  def apply(self, params):
    value = parse_value(self, params, value=self.value, operand="value")
    return params.data == value

class IsOneOfDatasetFilter(BaseDatasetFilter, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[DatasetFilterTypeEnum.IsOneOf]
  values: list[Any]
  def apply(self, params):
    values = list(map(lambda value: parse_value(self, params, value=value, operand="values"), self.values))
    return functools.reduce(
      lambda acc, cur: acc | (params.data == cur),
      values, np.full(len(params.data), 1, dtype=np.bool_)
    )

class GreaterThanDatasetFilter(BaseDatasetFilter, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[DatasetFilterTypeEnum.GreaterThan]
  value: Any
  def apply(self, params):
    value = parse_value(self, params, value=self.value, operand="value")
    return value > params.data
  
class LessThanDatasetFilter(BaseDatasetFilter, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[DatasetFilterTypeEnum.LessThan]
  value: Any
  def apply(self, params):
    value = parse_value(self, params, value=self.value, operand="value")
    return value < params.data
  
class GreaterThanOrEqualToDatasetFilter(BaseDatasetFilter, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[DatasetFilterTypeEnum.GreaterThanOrEqualTo]
  value: Any
  def apply(self, params):
    value = parse_value(self, params, value=self.value, operand="value")
    return value >= params.data
  
class LessThanOrEqualToDatasetFilter(BaseDatasetFilter, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[DatasetFilterTypeEnum.LessThanOrEqualTo]
  value: Any
  def apply(self, params):
    value = parse_value(self, params, value=self.value, operand="value")
    return value <= params.data

class HasTextDatasetFilter(BaseDatasetFilter, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[DatasetFilterTypeEnum.HasText]
  value: Any
  def apply(self, params):
    if params.column.type != SchemaColumnTypeEnum.Textual:
      raise DatasetFilterError.WrongColumnType(
        filter_type=self.type,
        column_type=params.column.type,
        target=self.target
      )
    return params.data.str.contains(self.value)

class BaseMulticategoricalDatasetFilter(BaseDatasetFilter, pydantic.BaseModel):
  def iterate(self, params: DatasetFilterParams):
    if params.column.type != SchemaColumnTypeEnum.MultiCategorical:
      raise DatasetFilterError.WrongColumnType(
        filter_type=self.type,
        column_type=params.column.type,
        target=self.target
      )
    mask = np.full(len(params.data), 1, dtype=np.bool_)
    yield mask
    for idx, json_tags in params.data:
      tags = json.loads(json_tags)
      mask[idx] = yield set(tags)

class IncludesDatasetFilter(BaseMulticategoricalDatasetFilter, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[DatasetFilterTypeEnum.Includes]
  values: list[str]
  def apply(self, params):
    coroutine = self.iterate(params)
    mask = cast(npt.NDArray, next(coroutine))
    values = set(map(str, self.values))
    for raw_tags in coroutine:
      tags = cast(list[str], raw_tags)
      coroutine.send(len(values.intersection(tags)) > 0)
    return mask

class ExcludesDatasetFilter(BaseMulticategoricalDatasetFilter, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[DatasetFilterTypeEnum.Excludes]
  values: list[str]
  def apply(self, params):
    coroutine = self.iterate(params)
    mask = cast(npt.NDArray, next(coroutine))
    values = set(map(str, self.values))
    for raw_tags in coroutine:
      tags = cast(list[str], raw_tags)
      coroutine.send(len(values.intersection(tags)) == 0)
    return mask

class OnlyDatasetFilter(BaseMulticategoricalDatasetFilter, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[DatasetFilterTypeEnum.Only]
  values: list[str]
  def apply(self, params):
    coroutine = self.iterate(params)
    mask = cast(npt.NDArray, next(coroutine))
    values = set(map(str, self.values))
    for raw_tags in coroutine:
      tags = cast(list[str], raw_tags)
      coroutine.send(len(values.symmetric_difference(tags)) == 0)
    return mask
  
__all__ = [
  "AndDatasetFilter",
  "OrDatasetFilter",
  "NotDatasetFilter",
  "EmptyDatasetFilter",
  "NotEmptyDatasetFilter",
  "EqualToDatasetFilter",
  "IsOneOfDatasetFilter",
  "GreaterThanDatasetFilter",
  "LessThanDatasetFilter",
  "GreaterThanOrEqualToDatasetFilter",
  "LessThanOrEqualToDatasetFilter",
  "HasTextDatasetFilter",
  "IncludesDatasetFilter",
  "ExcludesDatasetFilter",
  "OnlyDatasetFilter",
]