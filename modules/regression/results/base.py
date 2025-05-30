from enum import Enum
from typing import Optional
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
  target: str
  groups: list[NamedTableFilter]
  reference: Optional[str]
  interpretation: RegressionInterpretation
  constrain_by_groups: bool

  def assert_reference(self)->NamedTableFilter:
    if self.reference is None:
      raise MissingReferenceSubdatasetException()
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
  sample_size: int
  confidence_interval: tuple[float, float]
  variance_inflation_factor: float
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

class BaseRegressionResult(pydantic.BaseModel):
  model_id: str
  reference: Optional[str]
  interpretation: RegressionInterpretation
  converged: bool
  sample_size: int
  warnings: list[str]

class BaseRegressionPredictionInput(pydantic.BaseModel):
  model_id: str
  active: list[bool]
  def as_regression_input(self)->np.ndarray:
    return np.array([True, *self.active])

class BaseRegressionPredictionResult(pydantic.BaseModel):
  value: float