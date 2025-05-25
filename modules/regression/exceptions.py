from dataclasses import dataclass
from http import HTTPStatus
from modules.api.wrapper import ApiError, ApiErrorAdaptableException
from modules.table.filter_variants import NamedTableFilter

@dataclass
class ReferenceMustBeAValidSubdatasetException(ApiErrorAdaptableException):
  reference: str
  groups: list[str]

  def to_api(self):
    return ApiError(
      message=f"Reference \"{self.reference}\" does not exist in the subdatasets used as the independent variables for regression: {self.groups}.",
      status_code=HTTPStatus.NOT_FOUND
    )

  @staticmethod
  def assert_reference(reference: str, groups: list[NamedTableFilter]):
    group_names = list(map(lambda group: group.name, groups))
    for group in group_names:
      if reference == group:
        return
    raise ReferenceMustBeAValidSubdatasetException(
      reference=reference,
      groups=group_names,
    )
  
@dataclass
class NoIndependentVariableDataException(ApiErrorAdaptableException):
  column: str
  def to_api(self):
    return ApiError(
      message=f"Column \"{self.column}\" as the independent variable for the regression model does not contain any data that can be fitted to.",
      status_code=HTTPStatus.NOT_FOUND
    )
