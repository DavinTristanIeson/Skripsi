from dataclasses import dataclass
from http import HTTPStatus

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