import numpy as np
import pydantic

from modules.regression.results.base import BaseRegressionResult, RegressionCoefficient

class OrdinalRegressionCutpoint(pydantic.BaseModel):
  name: str
  value: float
  std_err: float

class OrdinalRegressionCoefficient(RegressionCoefficient, pydantic.BaseModel):
  @pydantic.computed_field
  def odds(self)->float:
    return np.exp(self.value)

class OrdinalRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  coefficients: list[OrdinalRegressionCoefficient]
  cutpoints: list[OrdinalRegressionCutpoint]
  log_likelihood_ratio: float
  converged: bool
  warnings: list[str]

__all__ = [
  "OrdinalRegressionResult"
]