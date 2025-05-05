from dataclasses import dataclass
from http import HTTPStatus
import os
from typing import Optional
from modules.api.wrapper import ApiError, ApiErrorAdaptableException

# To make it easier to catch exceptions
@dataclass
class FileLoadingException(ApiErrorAdaptableException):
  message: str
  def to_api(self):
    return ApiError(
      message=self.message,
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )

class FileNotExistsException(FileLoadingException):
  def to_api(self):
    return ApiError(
      message=self.message,
      status_code=HTTPStatus.NOT_FOUND
    )
  
  @staticmethod
  def format_message(
    *,
    path: str,
    purpose: str,
    problem: Optional[str] = None,
    solution: Optional[str] = None,
  ):
    return f"We were unable to find \"{path}\" to load the {purpose}." + (problem or '') + (solution or '')
  
  @staticmethod
  def verify(path:str, *, error: str):
    if not os.path.exists(path):
      raise FileNotExistsException(error)
  
@dataclass
class CorruptedFileException(FileLoadingException, ApiErrorAdaptableException):
  def to_api(self):
    return ApiError(
      message=self.message,
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )
  
  @staticmethod
  def format_message(
    *,
    path: str,
    purpose: str,
    solution: Optional[str] = None,
  ):
    return f"We were unable to read the {purpose} in \"{path}\". The file may have been corrupted or unintentionally modified." + (solution or '')