from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Sequence

import pandas as pd
import pydantic
from modules.api.wrapper import ApiError, ApiErrorAdaptableException
from modules.table.filter_variants import NamedTableFilter

class MissingReferenceSubdatasetException(ApiErrorAdaptableException):
  groups: list[str]
  def to_api(self):
    return ApiError(
      message=f"A subdataset must be chosen as the reference group if you want to interpret the regression results as relative to a reference. Pick one of the following subdatasets as your reference: {self.groups}.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )

@dataclass
class ReferenceMustBeAValidSubdatasetException(ApiErrorAdaptableException):
  reference: str
  groups: Sequence[str]

  def to_api(self):
    return ApiError(
      message=f"Reference \"{self.reference}\" does not exist in the subdatasets used as the independent variables for regression: {self.groups}.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )

  @staticmethod
  def assert_reference(reference: str, groups: list[NamedTableFilter])->NamedTableFilter:
    for group in groups:
      if reference == group.name:
        return group
    group_names = list(map(lambda group: group.name, groups))
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
  

class RegressionInterpretationRelativeToReferenceMutualExclusivityRequirementsViolationException(ApiErrorAdaptableException):
  def to_api(self):
    return ApiError(
      message=f"Your independent variables (subdatasets) are not mutually exclusive with each other. This means that we cannot naively set a variable as a reference since there may be interactions with other variables that could cause the model to produce inaccurate coefficients. Please change the interpretation to something else.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )
  
class RegressionInterpretationRelativeToBaselineMutualExclusivityRequirementsViolationException(ApiErrorAdaptableException):
  def to_api(self):
    return ApiError(
      message=f"There are no rows that can be used as the baseline for your independent variables (your subdatasets) as they are mutually exclusive with each other. We cannot naively perform regression otherwise we risk producing inflated/false coefficients due to multicollinearity. Please change the interpretation to something else.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )

class RegressionInterpretationGrandMeanDeviationMutualExclusivityRequirementsViolationException(ApiErrorAdaptableException):
  def to_api(self):
    return ApiError(
      message=f"Your independent variables (subdatasets) are not mutually exclusive from each other. This means that we cannot perform effect coding which is necessary to interpret regression results as deviation from the grand mean. Consider interpreting the regression as effects relative to a reference or a baseline instead.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )
    
class MissingLogisticRegressionDependentVariableReference(pydantic.BaseModel):
  def to_api(self):
    return ApiError(
      message=f"A reference category should be provided for the dependent variable of the multinomial logistic regression.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )
  
@dataclass
class DependentVariableReferenceMustBeAValidValueException(ApiErrorAdaptableException):
  reference: str
  supported_values: list[Any]

  def to_api(self):
    return ApiError(
      message=f"Reference \"{self.reference}\" is not a valid reference for the independent variable. Consider using one of the following instead: {list(map(str, self.supported_values))}. Otherwise, don't specify a reference so that we can pick a reference automatically.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )
