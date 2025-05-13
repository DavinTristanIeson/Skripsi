from dataclasses import dataclass
from http import HTTPStatus
from typing import Literal

from modules.api.wrapper import ApiError, ApiErrorAdaptableException


@dataclass
class RequiresTopicModelingException(ApiErrorAdaptableException):
  issue: str
  column: str
  def to_api(self):
    return ApiError(
      message=f"{self.issue}. Please run the topic modeling algorithm on {self.column} first.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )

@dataclass
class UnsyncedDocumentVectorsException(ApiErrorAdaptableException):
  type: str
  column: str
  observed_rows: int
  expected_rows: int
  def to_api(self):
    return ApiError(f"The topic modeling results are not in sync with the {self.type} for {self.column} (Found: {self.observed_rows}, Expected: {self.expected_rows}). The file may be corrupted. Try running the topic modeling procedure again.", HTTPStatus.BAD_REQUEST)
  
@dataclass
class MissingCachedTopicModelingResult(ApiErrorAdaptableException):
  type: str
  column: str
  def to_api(self):
    return ApiError(f"We were unable to find any {self.type} for {self.column} even though it should exist after the topic modeling algorithm has been executed. The file may be corrupted. Try running the topic modeling procedure again.", HTTPStatus.BAD_REQUEST)
  