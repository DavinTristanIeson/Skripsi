from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional
from modules.api.wrapper import ApiError, ApiErrorAdaptableException

@dataclass
class MissingColumnException(ApiErrorAdaptableException):
  message: str

  def to_api(self):
    return ApiError(
      message=self.message,
      status_code=HTTPStatus.NOT_FOUND
    )
  
  @staticmethod
  def format_schema_issue(column: str):
    return f"Column {column} doesn't exist in the schema. Please make sure that your schema is properly configured to your data."
  
@dataclass
class DataFrameLoadException(ApiErrorAdaptableException):
  message: str

  def to_api(self):
    return ApiError(
      message=self.message,
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )
  
  @staticmethod
  def format_message(path: str, purpose: str, solution: Optional[str], error: Exception):
    return f"Failed to load the {purpose} from {path}." + (solution or '') + f'Error: {error}'