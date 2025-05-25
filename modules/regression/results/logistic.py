from enum import Enum
from typing import Optional
import pydantic

from modules.api.enum import ExposedEnum
from modules.regression.results.base import BaseRegressionResult, RegressionCoefficient, RegressionIntercept
from modules.table.filter_variants import NamedTableFilter

class LogisticRegressionInterpretation(str, Enum):
  GrandMeanDeviation = "grand_mean_deviation"
  RelativeToReference = "relative_to_reference"
  RelativetoBaseline = "relative_to_baseline"

ExposedEnum().register(LogisticRegressionInterpretation)

class LogisticRegressionInput(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  reference: Optional[str]

class MultinomialLogisticRegressionInput(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  reference_independent: Optional[str]
  reference_dependent: Optional[str]

class LogisticRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  interpretation: LogisticRegressionInterpretation
  coefficients: list[RegressionCoefficient]
  intercept: RegressionIntercept
  statistic: float
  p_value: float
  pseudo_r_squared: float
  log_likelihood_ratio: float

class MultinomialLogisticRegressionFacetResult(pydantic.BaseModel):
  reference: str
  interpretation: LogisticRegressionInterpretation
  coefficients: list[RegressionCoefficient]
  intercept: RegressionIntercept

class MultinomialLogisticRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  facets: list[MultinomialLogisticRegressionFacetResult]
  chisq_statistic: float
  p_value: float
  pseudo_r_squared: float
  log_likelihood_ratio: float

  
__all__ = [
  "LogisticRegressionInput",
  "MultinomialLogisticRegressionInput",
  "LogisticRegressionResult",
  "MultinomialLogisticRegressionFacetResult",
  "MultinomialLogisticRegressionResult",
]