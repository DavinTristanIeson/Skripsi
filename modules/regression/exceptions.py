from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Sequence

import pandas as pd
import pydantic
from modules.api.wrapper import ApiError, ApiErrorAdaptableException
from modules.table.filter_variants import NamedTableFilter

@dataclass
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
      message=f"Your independent variables (subdatasets) are not mutually exclusive with each other. This means that we cannot naively set a variable as a reference since there may be interactions with other variables that could cause the model to produce inaccurate coefficients. Please change the interpretation to something else. Consider interpreting the regression as effects relative to the baseline instead.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )

class RegressionInterpretationRelativeToReferenceNotEnoughIndependentVariablesException(ApiErrorAdaptableException):
  def to_api(self):
    return ApiError(
      message=f"There should be at least two independent variables (subdatasets) for this interpretation; so that one independent variable will act as the reference.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )
  
class RegressionInterpretationRelativeToBaselineMutualExclusivityRequirementsViolationException(ApiErrorAdaptableException):
  def to_api(self):
    return ApiError(
      message=f"There are no rows that can be used as the baseline for your independent variables (your subdatasets) as they are mutually exclusive with each other. We cannot naively perform regression otherwise we risk producing inflated/false coefficients due to multicollinearity. Consider interpreting the regression as effects relative to a reference or as deviation from the grand mean instead.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )

class RegressionInterpretationGrandMeanDeviationMutualExclusivityRequirementsViolationException(ApiErrorAdaptableException):
  def to_api(self):
    return ApiError(
      message=f"Your independent variables (subdatasets) are not mutually exclusive from each other. This means that we cannot perform effect coding which is necessary to interpret regression results as deviation from the grand mean. Consider interpreting the regression as effects relative to the baseline instead.",
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

@dataclass
class MultilevelRegressionNotEnoughLevelsException(ApiErrorAdaptableException):
  type: str
  levels: Sequence[Any]
  def to_api(self):
    levels = list(map(str, self.levels))
    return ApiError(
      message=f"{self.type} regression typically involves dependent variables with more than two levels, but there are only {len(levels)} levels:  {'{' + ', '.join(levels) + '}'}]. If there are only 2 levels, consider using logistic regression instead.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )
  
  @staticmethod
  def assert_levels(type: str, levels: Sequence[Any]):
    if len(levels) <= 2:
      raise MultilevelRegressionNotEnoughLevelsException(
        type=type,
        levels=levels,
      )
    
RESERVED_SUBDATASET_NAMES = ["const", "Intercept", "Baseline"]
@dataclass
class ReservedSubdatasetNameException(ApiErrorAdaptableException):
  name: str
  def to_api(self):
    return ApiError(
      message=f"\"{self.name}\" is a reserved name. Please change the name of your subdataset.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )
  
  @staticmethod
  def assert_column_names(column_names: Sequence[str]):
    for name in RESERVED_SUBDATASET_NAMES:
      if name in column_names: 
        raise ReservedSubdatasetNameException(
          name=name,
        )
      
@dataclass
class MissingStoredRegressionModelException(ApiErrorAdaptableException):
  id: str
  def to_api(self):
    return ApiError(
      message=f"There are no stored regression models with this ID \"{self.id}\". This may be because the regression model has been deleted due to inactivity; please fit the regression model again.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )
  
@dataclass
class NonMutuallyExclusiveDependentVariableLevelsException(ApiErrorAdaptableException):
  def to_api(self):
    return ApiError(
      message=f"The levels of the dependent variable should be mutually exclusive.",
      status_code=HTTPStatus.UNPROCESSABLE_ENTITY
    )