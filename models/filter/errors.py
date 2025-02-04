import http
from types import SimpleNamespace
from typing import Any

from common.models.api import ApiError
from models.config.schema import SchemaColumnTypeEnum
from models.filter.types import DatasetFilterTypeEnum

class DatasetFilterError(SimpleNamespace):
  def WrongColumnType(*, filter_type: DatasetFilterTypeEnum, column_type: SchemaColumnTypeEnum, target: str):
    return ApiError(
      f"Filter of type {filter_type} is not compatible with {target} ({column_type} column)",
      http.HTTPStatus.UNPROCESSABLE_ENTITY
    )
  
  def WrongFieldValueType(*, type: DatasetFilterTypeEnum, target: str, value: Any, expected_type: str, operand_name: str, column_type: SchemaColumnTypeEnum):
    return ApiError(
      f"Filter of type {type} for {target} ({column_type} column) should have a {expected_type} for the operand \"{operand_name}\", but received \"{value}\" instead",
      http.HTTPStatus.UNPROCESSABLE_ENTITY
    )
  
__all__ = [
  "DatasetFilterError"
]