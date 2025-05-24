from dataclasses import dataclass
from http import HTTPStatus
from modules.api.wrapper import ApiError, ApiErrorAdaptableException
from modules.table.filter_variants import NamedTableFilter

@dataclass
class ReferenceMustBeAValidSubdataset(ApiErrorAdaptableException):
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
    raise ReferenceMustBeAValidSubdataset(
      reference=reference,
      groups=group_names,
    )