from dataclasses import dataclass
from http import HTTPStatus
from modules.api.wrapper import ApiError, ApiErrorAdaptableException


@dataclass
class UnallowedColumnOperationException(ApiErrorAdaptableException):
  column: str
  def to_api(self):
    return ApiError(
      message=f"Operations on the column \"{self.column}\" is not permitted since there's another process operating on this column at the moment",
      status_code=HTTPStatus.FORBIDDEN,
    )
  

@dataclass
class UnallowedFileOperationException(ApiErrorAdaptableException):
  path: str
  def to_api(self):
    return ApiError(
      message=f"Operations on the file \"{self.path}\" is not permitted since there's another process operating on the file at the moment",
      status_code=HTTPStatus.FORBIDDEN,
    )