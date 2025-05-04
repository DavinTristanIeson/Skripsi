from dataclasses import dataclass
import http
from types import SimpleNamespace
from typing import Any

from modules.api import ApiError
from modules.api.wrapper import ApiErrorAdaptableException
from modules.config import SchemaColumnTypeEnum

from .filter import TableFilterTypeEnum

@dataclass
class TableFilterWrongColumnTypeException(ApiErrorAdaptableException):
  filter_type: TableFilterTypeEnum
  column_type: SchemaColumnTypeEnum
  target: str

  def to_api(self):
    return ApiError(
      f"Filter of type {self.filter_type} is not compatible with \"{self.target}\" ({self.column_type} column)",
      http.HTTPStatus.UNPROCESSABLE_ENTITY
    )

@dataclass
class TableFilterWrongFieldValueTypeException(ApiErrorAdaptableException):
  type: TableFilterTypeEnum
  target: str
  value: Any
  expected_type: str
  operand_name: str
  column_type: SchemaColumnTypeEnum

  def to_api(self):
    return ApiError(
      f"Filter of type {self.type} for \"{self.target}\" ({self.column_type} column) should have a {self.expected_type} for the operand \"{self.operand_name}\", but received \"{self.value}\" instead",
      http.HTTPStatus.UNPROCESSABLE_ENTITY
    )

class TableFilterColumnNotFoundException(ApiErrorAdaptableException):
  target: str
  project_name: str
  def to_api(self):
    return ApiError(
      f"Filter target \"{self.target}\" does not exist in Project \"{self.project_name}\"",
      http.HTTPStatus.UNPROCESSABLE_ENTITY
    )
  