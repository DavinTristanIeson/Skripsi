import functools
import itertools
from typing import Annotated, Any, Literal, Optional, Sequence, Union, cast

import pandas as pd
import numpy as np
import pydantic

from modules.config import MultiCategoricalSchemaColumn, SchemaColumn, SchemaColumnTypeEnum
from modules.validation import DiscriminatedUnionValidator
from modules.logger import ProvisionedLogger

from .filter import _BaseCompoundTableFilter, _BaseTableFilter, _TableFilterParams, TableFilterTypeEnum
from .errors import _TableFilterError

logger = ProvisionedLogger().provision("TableEngine")

def access_series(filter: _BaseTableFilter, params: _TableFilterParams)->pd.Series:
  if filter.target not in params.data.columns:
    raise _TableFilterError.ColumnNotFound(
      target=filter.target,
      project_id=params.config.project_id
    )
  return params.data[filter.target]

ValueType = str | int | float
def parse_value(filter: _BaseTableFilter, params: _TableFilterParams, *, value: Any, operand: str)->Any: 
  column = params.config.data_schema.assert_exists(filter.target)
  ERROR_PAYLOAD = dict(
    type=filter.type,
    target=filter.target,
    operand_name=operand,
    value=value,
    column_type=column.type,
  )
  data = access_series(filter, params)
  if pd.api.types.is_numeric_dtype(data.dtype):
    try:
      return float(value) # type: ignore
    except ValueError:
      raise _TableFilterError.WrongFieldValueType(**ERROR_PAYLOAD, expected_type="numeric_value")
  elif pd.api.types.is_datetime64_any_dtype(data.dtype):
    try:
      return datetime.datetime.fromisoformat(value) # type: ignore
    except ValueError:
      raise _TableFilterError.WrongFieldValueType(**ERROR_PAYLOAD, expected_type="datetime string in ISO format")
  elif data.dtype == 'category':
    value = str(value)
    if str(value) not in data.cat.categories:
      raise _TableFilterError.WrongFieldValueType(**ERROR_PAYLOAD, expected_type="valid category")
    return value
  elif value is None:
    return None
  else:
    return str(value)
  
_AnyTableFilter = _BaseTableFilter | _BaseCompoundTableFilter
class AndTableFilter(_BaseCompoundTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.And] = TableFilterTypeEnum.And
  operands: list[_AnyTableFilter] = pydantic.Field(min_length=1)
  def __hash__(self):
    return hash(' '.join(itertools.chain(
      hex(hash(self.type)),
      map(lambda x: hex(hash(x)), self.operands)
    )))
  def apply(self, params):
    return functools.reduce(
      lambda acc, cur: acc & cur.apply(params),
      self.operands, np.full(len(params.data), 1, dtype=np.bool_)
    )
  
class OrTableFilter(_BaseCompoundTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.Or] = TableFilterTypeEnum.Or
  operands: list[_AnyTableFilter] = pydantic.Field(min_length=1)
  def __hash__(self):
    return hash(' '.join(itertools.chain(
      hex(hash(self.type)),
      map(lambda x: hex(hash(x)), self.operands)
    )))
  def apply(self, params):
    if len(self.operands) == 0:
      return np.full(len(params.data), 1, dtype=np.bool_)
    mask = self.operands[0].apply(params)
    for i in range(1, len(self.operands)):
      mask |= self.operands[i].apply(params)
    return mask

class NotTableFilter(_BaseCompoundTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.Not] = TableFilterTypeEnum.Not
  operand: _AnyTableFilter
  def __hash__(self):
    return hash(' '.join([
      hex(hash(self.type)),
      hex(hash(self.operand)),
    ]))
  
  def apply(self, params):
    return ~self.operand.apply(params)
  
class EmptyTableFilter(_BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.Empty] = TableFilterTypeEnum.Empty
  def apply(self, params):
    return access_series(self, params).isna()

class NotEmptyTableFilter(_BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.NotEmpty] = TableFilterTypeEnum.NotEmpty
  def apply(self, params):
    return access_series(self, params).notna()

class EqualToTableFilter(_BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.EqualTo] = TableFilterTypeEnum.EqualTo
  value: ValueType
  def apply(self, params):
    data = access_series(self, params)
    value = parse_value(self, params, value=self.value, operand="value")
    return data == value

class IsOneOfTableFilter(_BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.IsOneOf] = TableFilterTypeEnum.IsOneOf
  values: list[ValueType]
  def apply(self, params):
    data = access_series(self, params)
    values = list(map(lambda value: parse_value(self, params, value=value, operand="values"), self.values))
    return functools.reduce(
      lambda acc, cur: acc | (data == cur),
      values, np.full(len(data), 1, dtype=np.bool_)
    )

class GreaterThanTableFilter(_BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.GreaterThan] = TableFilterTypeEnum.GreaterThan
  value: ValueType
  def apply(self, params):
    data = access_series(self, params)
    value = parse_value(self, params, value=self.value, operand="value")
    return data > value
  
class LessThanTableFilter(_BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.LessThan] = TableFilterTypeEnum.LessThan
  value: ValueType
  def apply(self, params):
    data = access_series(self, params)
    value = parse_value(self, params, value=self.value, operand="value")
    return data < value
  
class GreaterThanOrEqualToTableFilter(_BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.GreaterThanOrEqualTo] = TableFilterTypeEnum.GreaterThanOrEqualTo
  value: ValueType
  def apply(self, params):
    data = access_series(self, params)
    value = parse_value(self, params, value=self.value, operand="value")
    return data >= value
  
class LessThanOrEqualToTableFilter(_BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.LessThanOrEqualTo] = TableFilterTypeEnum.LessThanOrEqualTo
  value: ValueType
  def apply(self, params):
    data = access_series(self, params)
    value = parse_value(self, params, value=self.value, operand="value")
    return data <= value

class HasTextTableFilter(_BaseTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.HasText] = TableFilterTypeEnum.HasText
  value: str
  def apply(self, params):
    data = access_series(self, params)
    column = params.config.data_schema.assert_exists(self.target)
    if column.type != SchemaColumnTypeEnum.Textual:
      raise _TableFilterError.WrongColumnType(
        filter_type=self.type,
        column_type=column.type,
        target=self.target
      )
    return data.str.contains(self.value)

class BaseMulticategoricalTableFilter(_BaseTableFilter, pydantic.BaseModel, frozen=True):
  def iterate(self, params: _TableFilterParams):
    column = params.config.data_schema.assert_exists(self.target)
    if column.type != SchemaColumnTypeEnum.MultiCategorical:
      raise _TableFilterError.WrongColumnType(
        filter_type=self.type,
        column_type=column.type,
        target=self.target
      )
    
    column = cast(MultiCategoricalSchemaColumn, column)
    data = cast(Sequence[str], access_series(self, params))
    mask = np.full(len(data), 1, dtype=np.bool_)
    yield mask
    for idx, tags in column.json2list(data):
      mask[idx] = yield set(tags)

class IncludesTableFilter(BaseMulticategoricalTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.Includes] = TableFilterTypeEnum.Includes
  values: list[str]
  def apply(self, params):
    coroutine = self.iterate(params)
    mask = cast(np.ndarray, next(coroutine))
    values = set(map(str, self.values))
    for raw_tags in coroutine:
      tags = cast(list[str], raw_tags)
      coroutine.send(len(values.intersection(tags)) > 0)
    return mask

class ExcludesTableFilter(BaseMulticategoricalTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.Excludes] = TableFilterTypeEnum.Excludes
  values: list[str]
  def apply(self, params):
    coroutine = self.iterate(params)
    mask = cast(np.ndarray, next(coroutine))
    values = set(map(str, self.values))
    for raw_tags in coroutine:
      tags = cast(list[str], raw_tags)
      coroutine.send(len(values.intersection(tags)) == 0)
    return mask

class OnlyTableFilter(BaseMulticategoricalTableFilter, pydantic.BaseModel, frozen=True):
  type: Literal[TableFilterTypeEnum.Only] = TableFilterTypeEnum.Only
  values: list[str]
  def apply(self, params):
    coroutine = self.iterate(params)
    mask = cast(np.ndarray, next(coroutine))
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
class NamedTableFilter(pydantic.BaseModel):
  name: str
  filter: TableFilter

  
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
  "NamedTableFilter"
]