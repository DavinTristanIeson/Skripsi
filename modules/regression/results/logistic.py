from enum import Enum
from typing import Optional
import pydantic

from modules.regression.results.base import BaseRegressionInput, BaseRegressionResult, RegressionCoefficient, RegressionInterpretation

class MultinomialLogisticRegressionInput(BaseRegressionInput, pydantic.BaseModel):
  reference_dependent: Optional[str]

class LogisticRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  coefficients: list[RegressionCoefficient]
  intercept: RegressionCoefficient
  p_value: float
  pseudo_r_squared: float
  log_likelihood_ratio: float

class OneVsRestLogisticRegressionResult(pydantic.BaseModel):
  results: list[LogisticRegressionResult]

class MultinomialLogisticRegressionFacetResult(pydantic.BaseModel):
  coefficients: list[RegressionCoefficient]
  intercept: RegressionCoefficient

class MultinomialLogisticRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  reference: Optional[str]
  facets: list[MultinomialLogisticRegressionFacetResult]
  p_value: float
  pseudo_r_squared: float
  log_likelihood_ratio: float

  
__all__ = [
  "MultinomialLogisticRegressionInput",
  "LogisticRegressionResult",
  "MultinomialLogisticRegressionFacetResult",
  "MultinomialLogisticRegressionResult",
  "OneVsRestLogisticRegressionResult",
]