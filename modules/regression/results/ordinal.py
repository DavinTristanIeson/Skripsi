from enum import Enum
from typing import Optional
import pydantic

from modules.api.enum import ExposedEnum
from modules.regression.results.base import RegressionCoefficient, RegressionIntercept
from modules.table.filter_variants import NamedTableFilter

class OrdinalRegressionInterpretation:
  GrandMeanDeviation = "grand_mean_deviation"
  RelativeToReference = "relative_to_reference"
  RelativetoBaseline = "relative_to_baseline"

ExposedEnum().register(OrdinalRegressionInterpretation)

class OrdinalRegressionInput(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  reference: Optional[str]
  with_intercept: bool
  standardized: bool

class OrdinalRegressionResult(pydantic.BaseModel):
  reference: Optional[str]
  interpretation: OrdinalRegressionInterpretation
  coefficients: list[RegressionCoefficient]
  intercept: Optional[RegressionIntercept]
  log_likelihood_ratio: float
  converged: bool
  warnings: list[str]

__all__ = [
  "OrdinalRegressionInterpretation",
  "OrdinalRegressionInput",
  "OrdinalRegressionResult"
]