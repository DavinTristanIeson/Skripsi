from dataclasses import dataclass
from http import HTTPStatus
from typing import Iterable, Literal, Sequence
from modules.api.wrapper import ApiError, ApiErrorAdaptableException


@dataclass
class MissingColumnInSchemaException(ApiErrorAdaptableException):
  column: str
  def to_api(self) -> ApiError:
    return ApiError(
      message=f"Column \"{self.column}\" doesn't exist in the schema",
      status_code=HTTPStatus.NOT_FOUND
    )

@dataclass
class WrongSchemaColumnTypeException(ApiErrorAdaptableException):
  column: str
  column_type: str
  types: Sequence[str]
  def to_api(self) -> ApiError:
    return ApiError(
      message=f"The type of column \"{self.column}\" ({self.column_type}) is not one of these types: {', '.join(self.types)}",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )

@dataclass
class DataFrameColumnFitException(ApiErrorAdaptableException):
  column: str
  error: Exception
  def to_api(self) -> ApiError:
    return ApiError(
      message=f"An error has occurred while fitting \"{self.column}\": {self.error}. Please check if your dataset matches your column configuration.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )

@dataclass
class UnsyncedDatasetSchemaException(ApiErrorAdaptableException):
  columns: Iterable[str]
  def to_api(self) -> ApiError:
    return ApiError(
      message=f"The columns specified in the project configuration does not match the column in the dataset. Please remove these columns ({' '.join(list(self.columns))}) from the project configuration if they do not exist in the dataset",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )

@dataclass
class MissingTextualColumnInternalColumnsException(ApiErrorAdaptableException):
  mode: Literal['preprocess', 'topics']
  def to_api(self):
    SHARED = f"If you haven't run the topic modeling algorithm before, please run the topic modeling algorithm first from the \"Topics\" page. If you have already executed the topic modeling procedure before, it is likely that the topic-related files are missing or corrupted."
    if self.mode == 'preprocess':
      return ApiError(
        message=f"There are no cached preprocessed documents in the dataframe. {SHARED}",
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY
      )
    elif self.mode == 'topics':
      return ApiError(
        message=f"There are no cached topics in the dataframe. Please run the topic modeling algorithm first. {SHARED}",
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY
      )
    else:
      raise ValueError(f"Invalid MissingTextualColumnInternalColumnsException mode: {self.mode}")