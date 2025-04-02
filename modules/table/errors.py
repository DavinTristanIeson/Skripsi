import http
from types import SimpleNamespace
from typing import Any

from modules.api import ApiError
from modules.config import SchemaColumnTypeEnum

from .filter import TableFilterTypeEnum

class _TableFilterError(SimpleNamespace):
  def WrongColumnType(*, filter_type: TableFilterTypeEnum, column_type: SchemaColumnTypeEnum, target: str):
    return ApiError(
      f"Filter of type {filter_type} is not compatible with \"{target}\" ({column_type} column)",
      http.HTTPStatus.UNPROCESSABLE_ENTITY
    )
  
  def WrongFieldValueType(*, type: TableFilterTypeEnum, target: str, value: Any, expected_type: str, operand_name: str, column_type: SchemaColumnTypeEnum):
    return ApiError(
      f"Filter of type {type} for \"{target}\" ({column_type} column) should have a {expected_type} for the operand \"{operand_name}\", but received \"{value}\" instead",
      http.HTTPStatus.UNPROCESSABLE_ENTITY
    )
  
  def ColumnNotFound(*, target: str, project_name: str):
    return ApiError(
      f"Filter target \"{target}\" does not exist in Project \"{project_name}\"",
      http.HTTPStatus.UNPROCESSABLE_ENTITY
    )
  