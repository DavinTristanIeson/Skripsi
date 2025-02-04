from typing import Any
import pandas as pd

from models.filter.errors import DatasetFilterError
from models.filter.types import BaseDatasetFilter, DatasetFilterParams

def parse_value(filter: BaseDatasetFilter, params: DatasetFilterParams, *, value: Any, operand: str)->Any: 
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
      raise DatasetFilterError.WrongFieldValueType(**ERROR_PAYLOAD, expected_type="numeric_value")
  elif pd.api.types.is_datetime64_any_dtype(data.dtype):
    try:
      return datetime.datetime.fromisoformat(value) # type: ignore
    except ValueError:
      raise DatasetFilterError.WrongFieldValueType(**ERROR_PAYLOAD, expected_type="datetime string in ISO format")
  elif data.dtype == 'category':
    value = str(value)
    if str(value) not in data.cat.categories:
      raise DatasetFilterError.WrongFieldValueType(**ERROR_PAYLOAD, expected_type="valid category")
    return value
  elif value is None:
    return None
  else:
    return str(value)