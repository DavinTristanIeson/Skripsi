from enum import Enum
from typing import Generic, Optional, TypeVar
import numpy as np
import pydantic

from modules.api.enum import ExposedEnum
from modules.regression.exceptions import MissingReferenceSubdatasetException, ReferenceMustBeAValidSubdatasetException
from modules.table.filter_variants import NamedTableFilter

class RegressionInterpretation(str, Enum):
  GrandMeanDeviation = "grand_mean_deviation"
  RelativeToReference = "relative_to_reference"
  RelativeToBaseline = "relative_to_baseline"

ExposedEnum().register(RegressionInterpretation)

class BaseRegressionInput(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  reference: Optional[str]
  interpretation: RegressionInterpretation
  constrain_by_groups: bool

  def assert_reference(self)->NamedTableFilter:
    if self.reference is None:
      raise MissingReferenceSubdatasetException(groups=list(map(lambda group: group.name, self.groups)))
    return ReferenceMustBeAValidSubdatasetException.assert_reference(self.reference, self.groups)
  
  @pydantic.model_validator(mode="after")
  def __ensure_reference_exists(self):
    if self.interpretation == RegressionInterpretation.RelativeToReference:
      self.assert_reference()
    return self

class RegressionCoefficient(pydantic.BaseModel):
  name: str
  value: float
  p_value: float
  std_err: float
  confidence_interval: tuple[float, float]
  statistic: float

class OddsBasedRegressionCoefficient(RegressionCoefficient, pydantic.BaseModel):
  @pydantic.computed_field
  def odds_ratio(self)->float:
    return np.exp(self.value)
  
  @pydantic.computed_field
  def odds_ratio_confidence_interval(self)->tuple[float, float]:
    return (
      np.exp(self.confidence_interval[0]),
      np.exp(self.confidence_interval[1])
    )

class RegressionIndependentVariableInfo(pydantic.BaseModel):
  name: str
  sample_size: int
  variance_inflation_factor: float

class BaseRegressionResult(pydantic.BaseModel):
  model_id: str
  independent_variables: list[RegressionIndependentVariableInfo]
  reference: Optional[str]

  interpretation: RegressionInterpretation
  sample_size: int

  warnings: list[str]

class BaseRegressionFitEvaluationResult(pydantic.BaseModel):
  converged: bool
  p_value: float
  model_dof: float
  residual_dof: float

class LogLikelihoodBasedFitEvaluation(BaseRegressionFitEvaluationResult, pydantic.BaseModel):
  log_likelihood_ratio: float
  log_likelihood: float
  log_likelihood_null: float
  aic: float
  bic: float
  pseudo_r_squared: float

  @pydantic.computed_field
  def likelihood_ratio(self)->float:
    return np.exp(self.log_likelihood_ratio / 2)


T = TypeVar("T")
class RegressionPredictionPerIndependentVariableResult(pydantic.BaseModel, Generic[T]):
  variable: str
  prediction: T

class BaseRegressionPredictionInput(pydantic.BaseModel):
  model_id: str
  input: list[float]
  def as_regression_input(self)->np.ndarray:
    return np.array([1.0, *self.input])

class RegressionDependentVariableLevelInfo(pydantic.BaseModel):
  name: str
  sample_size: int