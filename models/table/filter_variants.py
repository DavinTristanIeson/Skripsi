import functools
import json
from typing import Annotated, Any, Literal, Union, cast

import numpy as np
import pydantic
from models.config.schema import MultiCategoricalSchemaColumn, SchemaColumnTypeEnum
from models.table.errors import TableFilterError
from models.table.filter import BaseTableFilter, TableFilterParams, TableFilterTypeEnum
from common.models.validators import CommonModelConfig, DiscriminatedUnionValidator
from common.logger import RegisteredLogger
import numpy.typing as npt
import pandas as pd

logger = RegisteredLogger().provision("Filter Controller")

def parse_value(filter: BaseTableFilter, params: TableFilterParams, *, value: Any, operand: str)->Any: 
  ERROR_PAYLOAD = dict(
    type=filter.type,
    target=filter.target,
    operand_name=operand,
    value=value,
    column_type=params.column.type,
  )
  data = params.data
  if pd.api.types.is_numeric_dtype(data.dtype):
    try:
      return float(value) # type: ignore
    except ValueError:
      raise TableFilterError.WrongFieldValueType(**ERROR_PAYLOAD, expected_type="numeric_value")
  elif pd.api.types.is_datetime64_any_dtype(data.dtype):
    try:
      return datetime.datetime.fromisoformat(value) # type: ignore
    except ValueError:
      raise TableFilterError.WrongFieldValueType(**ERROR_PAYLOAD, expected_type="datetime string in ISO format")
  elif data.dtype == 'category':
    value = str(value)
    if str(value) not in data.cat.categories:
      raise TableFilterError.WrongFieldValueType(**ERROR_PAYLOAD, expected_type="valid category")
    return value
  elif value is None:
    return None
  else:
    return str(value)
  
class AndTableFilter(BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.And]
  operands: list[BaseTableFilter] = pydantic.Field(min_length=1)
  def apply(self, params):
    return functools.reduce(
      lambda acc, cur: acc & cur.apply(params),
      self.operands, np.full(len(params.data), 1, dtype=np.bool_)
    )
  
class OrTableFilter(BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.Or]
  operands: list[BaseTableFilter] = pydantic.Field(min_length=1)
  def apply(self, params):
    return functools.reduce(
      lambda acc, cur: acc | cur.apply(params),
      self.operands, np.full(len(params.data), 1, dtype=np.bool_)
    )

class NotTableFilter(BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.Not]
  operand: BaseTableFilter
  def apply(self, params):
    return ~self.operand.apply(params)
  
class EmptyTableFilter(BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.Empty]
  def apply(self, params):
    return params.data.isna()

class NotEmptyTableFilter(BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.NotEmpty]
  def apply(self, params):
    return params.data.notna()

class EqualToTableFilter(BaseTableFilter, pydantic.BaseModel, frozen=True):
  model_config = CommonModelConfig
  type: Literal[TableFilterTypeEnum.EqualTo]
  value: str
  def apply(self, params):
    value = parse_value(self, params, value=self.value, operand="value")
    return params.data == value

class IsOneOfTableFilter(BaseTableFilter, pydantic.BaseModel, frozen=True):
  model_config = CommonModelConfig
  type: Literal[TableFilterTypeEnum.IsOneOf]
  values: list[str]
  def apply(self, params):
    values = list(map(lambda value: parse_value(self, params, value=value, operand="values"), self.values))
    return functools.reduce(
      lambda acc, cur: acc | (params.data == cur),
      values, np.full(len(params.data), 1, dtype=np.bool_)
    )

class GreaterThanTableFilter(BaseTableFilter, pydantic.BaseModel, frozen=True):
  model_config = CommonModelConfig
  type: Literal[TableFilterTypeEnum.GreaterThan]
  value: str
  def apply(self, params):
    value = parse_value(self, params, value=self.value, operand="value")
    return value > params.data
  
class LessThanTableFilter(BaseTableFilter, pydantic.BaseModel, frozen=True):
  model_config = CommonModelConfig
  type: Literal[TableFilterTypeEnum.LessThan]
  value: str
  def apply(self, params):
    value = parse_value(self, params, value=self.value, operand="value")
    return value < params.data
  
class GreaterThanOrEqualToTableFilter(BaseTableFilter, pydantic.BaseModel, frozen=True):
  model_config = CommonModelConfig
  type: Literal[TableFilterTypeEnum.GreaterThanOrEqualTo]
  value: str
  def apply(self, params):
    value = parse_value(self, params, value=self.value, operand="value")
    return value >= params.data
  
class LessThanOrEqualToTableFilter(BaseTableFilter, pydantic.BaseModel, frozen=True):
  model_config = CommonModelConfig
  type: Literal[TableFilterTypeEnum.LessThanOrEqualTo]
  value: str
  def apply(self, params):
    value = parse_value(self, params, value=self.value, operand="value")
    return value <= params.data

class HasTextTableFilter(BaseTableFilter, pydantic.BaseModel, frozen=True):
  model_config = CommonModelConfig
  type: Literal[TableFilterTypeEnum.HasText]
  value: str
  def apply(self, params):
    if params.column.type != SchemaColumnTypeEnum.Textual:
      raise TableFilterError.WrongColumnType(
        filter_type=self.type,
        column_type=params.column.type,
        target=self.target
      )
    return params.data.str.contains(self.value)

class BaseMulticategoricalTableFilter(BaseTableFilter, pydantic.BaseModel, frozen=True):
  def iterate(self, params: TableFilterParams):
    if params.column.type != SchemaColumnTypeEnum.MultiCategorical:
      raise TableFilterError.WrongColumnType(
        filter_type=self.type,
        column_type=params.column.type,
        target=self.target
      )
    mask = np.full(len(params.data), 1, dtype=np.bool_)
    yield mask
    for idx, json_tags in params.data:
      tags = json.loads(json_tags)
      mask[idx] = yield set(tags)

class IncludesTableFilter(BaseMulticategoricalTableFilter, pydantic.BaseModel, frozen=True):
  model_config = CommonModelConfig
  type: Literal[TableFilterTypeEnum.Includes]
  values: list[str]
  def apply(self, params):
    coroutine = self.iterate(params)
    mask = cast(npt.NDArray, next(coroutine))
    values = set(map(str, self.values))
    for raw_tags in coroutine:
      tags = cast(list[str], raw_tags)
      coroutine.send(len(values.intersection(tags)) > 0)
    return mask

class ExcludesTableFilter(BaseMulticategoricalTableFilter, pydantic.BaseModel, frozen=True):
  model_config = CommonModelConfig
  type: Literal[TableFilterTypeEnum.Excludes]
  values: list[str]
  def apply(self, params):
    coroutine = self.iterate(params)
    mask = cast(npt.NDArray, next(coroutine))
    values = set(map(str, self.values))
    for raw_tags in coroutine:
      tags = cast(list[str], raw_tags)
      coroutine.send(len(values.intersection(tags)) == 0)
    return mask

class OnlyTableFilter(BaseMulticategoricalTableFilter, pydantic.BaseModel, frozen=True):
  model_config = CommonModelConfig
  type: Literal[TableFilterTypeEnum.Only]
  values: list[str]
  def apply(self, params):
    coroutine = self.iterate(params)
    mask = cast(npt.NDArray, next(coroutine))
    values = set(map(str, self.values))
    for raw_tags in coroutine:
      tags = cast(list[str], raw_tags)
      coroutine.send(len(values.symmetric_difference(tags)) == 0)
    return mask
  
TableFilterUnion = Union[
  AndTableFilter,
  OrTableFilter,
  NotTableFilter,
  EmptyTableFilter,
  NotEmptyTableFilter,
  EqualToTableFilter,
  IsOneOfTableFilter,
  GreaterThanTableFilter,
  LessThanTableFilter,
  GreaterThanOrEqualToTableFilter,
  LessThanOrEqualToTableFilter,
  HasTextTableFilter,
  IncludesTableFilter,
  ExcludesTableFilter,
  OnlyTableFilter,
]
TableFilter = Annotated[TableFilterUnion, pydantic.Field(discriminator="type"), DiscriminatedUnionValidator]
  
__all__ = [
  "AndTableFilter",
  "OrTableFilter",
  "NotTableFilter",
  "EmptyTableFilter",
  "NotEmptyTableFilter",
  "EqualToTableFilter",
  "IsOneOfTableFilter",
  "GreaterThanTableFilter",
  "LessThanTableFilter",
  "GreaterThanOrEqualToTableFilter",
  "LessThanOrEqualToTableFilter",
  "HasTextTableFilter",
  "IncludesTableFilter",
  "ExcludesTableFilter",
  "OnlyTableFilter",
  "TableFilter",
]