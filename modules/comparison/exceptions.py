from dataclasses import dataclass
from http import HTTPStatus
from typing import Sequence
from modules.api.wrapper import ApiError, ApiErrorAdaptableException
  
@dataclass
class NotEnoughComparisonGroupsException(ApiErrorAdaptableException):
  def to_api(self) -> ApiError:
    return ApiError(
      message=f"At least two groups has to be provided for a statistic test.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )

@dataclass
class EmptyComparisonGroupException(ApiErrorAdaptableException):
  group: str
  def to_api(self) -> ApiError:
    return ApiError(
      message=f"{self.group} does not have any values that can be compared.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )

@dataclass
class InvalidColumnTypeForComparisonMethodException(ApiErrorAdaptableException):
  method: str
  supported_types: Sequence[str]
  column_type: str
  def to_api(self):
    return ApiError(
      message=f"{self.method} can only be used to compare columns of type {', '.join(self.supported_types)}, but received \"{self.column_type}\" instead.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )
