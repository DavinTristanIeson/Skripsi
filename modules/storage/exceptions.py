from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional
from modules.api.wrapper import ApiError, ApiErrorAdaptableException

@dataclass
class FileSystemCleanupError(ApiErrorAdaptableException):
  path: str
  error: Exception
  def to_api(self):
    return ApiError(
      message=f"An unexpected error has occurred while cleaning up \"{self.path}\": {self.error}",
      status_code=HTTPStatus.INTERNAL_SERVER_ERROR
    )
  
@dataclass
class UserDataUniquenessConstraintViolationException(ApiErrorAdaptableException):
  id: Optional[str]
  name: Optional[str]
  def __post_init__(self):
    if self.id is None and self.name is None:
      raise ValueError(f"ID or name must be provided to {self.__class__.__name__}")
    
  def to_api(self):
    if self.id is not None:  
      return ApiError(
        message=f"ID \"{self.id}\" already exists.",
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY
      )
    elif self.name is not None:
      return ApiError(
        message=f"The name \"{self.name}\" already exists. Please choose another name.",
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY
      )
    else:
      raise ValueError(f"ID or name must be provided to {self.__class__.__name__}")
  
@dataclass
class UserDataEntryNotFoundException(ApiErrorAdaptableException):
  id: str
  path: str
  def to_api(self):
    return ApiError(
      message=f"There are no entries with ID \"{self.id}\" in \"{self.path}\"",
      status_code=HTTPStatus.NOT_FOUND
    )
