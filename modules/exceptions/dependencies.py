from dataclasses import dataclass
from http import HTTPStatus
from typing import Iterable, Sequence

from modules.api.wrapper import ApiError, ApiErrorAdaptableException

@dataclass
class DependencyImportException(ApiErrorAdaptableException):
  name: str
  purpose: str
  def to_api(self):
    return ApiError(
      message=f"The {self.name} library must be installed before {self.purpose}. Please ensure that you have set up the application correctly.",
      status_code=HTTPStatus.INTERNAL_SERVER_ERROR
    )

@dataclass
class InvalidValueTypeException(ApiErrorAdaptableException):
  value: str
  type: str

  def to_api(self):
    return ApiError(
      message=f"\"{self.value}\" is not a valid {self.type}.",
      status_code=HTTPStatus.INTERNAL_SERVER_ERROR
    )