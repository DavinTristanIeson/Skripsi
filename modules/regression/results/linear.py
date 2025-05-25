from enum import Enum
from typing import Optional
import pydantic

from modules.api.enum import ExposedEnum
from modules.regression.results.base import BaseRegressionResult, RegressionCoefficient, RegressionIntercept
from modules.table.filter_variants import NamedTableFilter

class LinearRegressionInterpretation(str, Enum):
  GrandMeanDeviation = "grand_mean_deviation"
  RelativeToReference = "relative_to_reference"
  RelativetoBaseline = "relative_to_baseline"

ExposedEnum().register(LinearRegressionInterpretation)

class LinearRegressionInput(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  reference: Optional[str]
  with_intercept: bool
  standardized: bool

class LinearRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  interpretation: LinearRegressionInterpretation
  coefficients: list[RegressionCoefficient]
  intercept: Optional[RegressionIntercept]
  f_statistic: float
  p_value: float
  r_squared: float

__all__ = [
  "LinearRegressionInterpretation",
  "LinearRegressionInput",
  "LinearRegressionResult"
]