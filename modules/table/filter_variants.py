import datetime
from typing import Annotated, Any, Literal, Union

import numpy as np
import pandas as pd
import pydantic

from modules.config import SchemaColumnTypeEnum
from modules.validation import DiscriminatedUnionValidator
from modules.logger import ProvisionedLogger

from .filter import _BaseCompoundTableFilter, _BaseTableFilter, _TableFilterParams, TableFilterTypeEnum
from .errors import TableFilterColumnNotFoundException, TableFilterWrongColumnTypeException, TableFilterWrongFieldValueTypeException

logger = ProvisionedLogger().provision("TableEngine")

def access_series(filter: _BaseTableFilter, params: _TableFilterParams)->pd.Series:
  if filter.target not in params.data.columns:
    raise TableFilterColumnNotFoundException(
      target=filter.target,
      project_name=params.config.metadata.name,
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
      if data.dtype == "Int32" or data.dtype == "Int64":
        return int(value)
      return float(value)
    except ValueError:
      raise TableFilterWrongFieldValueTypeException(**ERROR_PAYLOAD, expected_type="numeric_value")
  elif pd.api.types.is_datetime64_any_dtype(data.dtype):
    try:
      # Python fromisoformat doesn't recognize Z.
      value = str(value).replace("Z", "+00:00")
      date_value = np.datetime64(value)
      if np.isnat(date_value):
        raise TableFilterWrongFieldValueTypeException(**ERROR_PAYLOAD, expected_type="a valid datetime string in ISO format")
      return date_value
    except ValueError:
      raise TableFilterWrongFieldValueTypeException(**ERROR_PAYLOAD, expected_type="datetime string in ISO format")
  elif data.dtype == 'category':
    value = str(value)
    if str(value) not in data.cat.categories:
      raise TableFilterWrongFieldValueTypeException(**ERROR_PAYLOAD, expected_type=f"valid category (one of {data.cat.categories})")
    return value
  elif value is None:
    return None
  else:
    return str(value)
  

class AndTableFilter(_BaseCompoundTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.And] = TableFilterTypeEnum.And
  operands: list["TableFilter"]
  def apply(self, params):
    mask = params.mask(True)
    for operand in self.operands:
      new_mask = operand.apply(params)
      mask = mask & new_mask
    return mask
  
class OrTableFilter(_BaseCompoundTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.Or] = TableFilterTypeEnum.Or
  operands: list["TableFilter"]
  def apply(self, params):
    if len(self.operands) == 0:
      return params.mask(True)
    mask = params.mask(False)
    for operand in self.operands:
      new_mask = operand.apply(params)
      mask = mask | new_mask
    return mask

class NotTableFilter(_BaseCompoundTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.Not] = TableFilterTypeEnum.Not
  operand: "TableFilter"
  def apply(self, params):
    return ~self.operand.apply(params)
  
class EmptyTableFilter(_BaseTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.Empty] = TableFilterTypeEnum.Empty
  def apply(self, params):
    return access_series(self, params).isna()

class NotEmptyTableFilter(_BaseTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.NotEmpty] = TableFilterTypeEnum.NotEmpty
  def apply(self, params):
    return access_series(self, params).notna()

class EqualToTableFilter(_BaseTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.EqualTo] = TableFilterTypeEnum.EqualTo
  value: ValueType
  def apply(self, params):
    data = access_series(self, params)
    value = parse_value(self, params, value=self.value, operand="value")
    return (data == value).fillna(False)
  
class IsOneOfTableFilter(_BaseTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.IsOneOf] = TableFilterTypeEnum.IsOneOf
  values: list[ValueType]
  def apply(self, params):
    data = access_series(self, params)
    values = list(map(lambda value: parse_value(self, params, value=value, operand="values"), self.values))
    mask = params.mask(False)
    for value in values:
      new_mask = (data == value).fillna(False)
      mask = mask | new_mask
    return mask

class GreaterThanTableFilter(_BaseTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.GreaterThan] = TableFilterTypeEnum.GreaterThan
  value: ValueType
  def apply(self, params):
    data = access_series(self, params)
    value = parse_value(self, params, value=self.value, operand="value")
    return (data > value).fillna(False)
  
class LessThanTableFilter(_BaseTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.LessThan] = TableFilterTypeEnum.LessThan
  value: ValueType
  def apply(self, params):
    data = access_series(self, params)
    value = parse_value(self, params, value=self.value, operand="value")
    return (data < value).fillna(False)
  
class GreaterThanOrEqualToTableFilter(_BaseTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.GreaterThanOrEqualTo] = TableFilterTypeEnum.GreaterThanOrEqualTo
  value: ValueType
  def apply(self, params):
    data = access_series(self, params)
    value = parse_value(self, params, value=self.value, operand="value")
    return (data >= value).fillna(False)
  
class LessThanOrEqualToTableFilter(_BaseTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.LessThanOrEqualTo] = TableFilterTypeEnum.LessThanOrEqualTo
  value: ValueType
  def apply(self, params):
    data = access_series(self, params)
    value = parse_value(self, params, value=self.value, operand="value")
    return (data <= value).fillna(False)

class HasTextTableFilter(_BaseTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.HasText] = TableFilterTypeEnum.HasText
  value: str
  def apply(self, params):
    data = access_series(self, params)
    column = params.config.data_schema.assert_exists(self.target)
    if column.type not in [SchemaColumnTypeEnum.Textual, SchemaColumnTypeEnum.Categorical, SchemaColumnTypeEnum.OrderedCategorical, SchemaColumnTypeEnum.Unique]:
      raise TableFilterWrongColumnTypeException(
        filter_type=self.type,
        column_type=column.type,
        target=self.target
      )
    return data.str.contains(self.value).fillna(False)
  
class IsTrueTableFilter(_BaseTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.IsTrue] = TableFilterTypeEnum.IsTrue
  def apply(self, params):
    data = access_series(self, params)
    return (data == True).fillna(False)

class IsFalseTableFilter(_BaseTableFilter, pydantic.BaseModel):
  type: Literal[TableFilterTypeEnum.IsFalse] = TableFilterTypeEnum.IsFalse
  def apply(self, params):
    data = access_series(self, params)
    return (data == False).fillna(False)
  
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
  IsTrueTableFilter,
  IsFalseTableFilter
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
  "IsTrueTableFilter",
  "IsFalseTableFilter",
  "TableFilter",
  "NamedTableFilter"
]