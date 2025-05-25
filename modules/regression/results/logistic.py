from enum import Enum
from typing import Optional
import pydantic

from modules.regression.results.base import BaseRegressionInput, BaseRegressionResult, RegressionCoefficient, RegressionInterpretation

class MultinomialLogisticRegressionType(str, Enum):
  Individual = "individual"
  Full = "full"

class MultinomialLogisticRegressionInput(BaseRegressionInput, pydantic.BaseModel):
  type: MultinomialLogisticRegressionType
  reference_dependent: Optional[str]

class LogisticRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  interpretation: RegressionInterpretation
  coefficients: list[RegressionCoefficient]
  intercept: RegressionCoefficient
  p_value: float
  pseudo_r_squared: float
  log_likelihood_ratio: float

class IndividualLogisticRegressionMainResult(pydantic.BaseModel):
  results: list[LogisticRegressionResult]

class MultinomialLogisticRegressionFacetResult(pydantic.BaseModel):
  reference: str
  interpretation: RegressionInterpretation
  coefficients: list[RegressionCoefficient]
  intercept: RegressionCoefficient

class MultinomialLogisticRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  facets: list[MultinomialLogisticRegressionFacetResult]
  chisq_statistic: float
  p_value: float
  pseudo_r_squared: float
  log_likelihood_ratio: float

  
__all__ = [
  "MultinomialLogisticRegressionInput",
  "LogisticRegressionResult",
  "MultinomialLogisticRegressionFacetResult",
  "MultinomialLogisticRegressionResult",
  "IndividualLogisticRegressionMainResult",
]