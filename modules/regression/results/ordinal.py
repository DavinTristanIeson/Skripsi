from typing import Optional
import pydantic

from modules.regression.results.base import BaseRegressionResult, RegressionCoefficient, RegressionInterpretation

class OrdinalRegressionCutpoint(pydantic.BaseModel):
  name: str
  value: float
  std_err: float

class OrdinalRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  coefficients: list[RegressionCoefficient]
  cutpoints: list[OrdinalRegressionCutpoint]
  log_likelihood_ratio: float
  converged: bool
  warnings: list[str]

__all__ = [
  "OrdinalRegressionResult"
]