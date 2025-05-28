import numpy as np
import pydantic

from modules.regression.results.base import BaseRegressionResult, RegressionCoefficient

class OrdinalRegressionCutpoint(pydantic.BaseModel):
  name: str
  value: float
  std_err: float
  confidence_interval: tuple[float, float]
  sample_size: int

class OrdinalRegressionCoefficient(RegressionCoefficient, pydantic.BaseModel):
  @pydantic.computed_field
  def odds_ratio(self)->float:
    # Odds of being in higher ranks compared to lower ranks
    return np.exp(-self.value)

class OrdinalRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  coefficients: list[OrdinalRegressionCoefficient]
  cutpoints: list[OrdinalRegressionCutpoint]
  log_likelihood_ratio: float
  p_value: float
  pseudo_r_squared: float
  converged: bool
  warnings: list[str]

__all__ = [
  "OrdinalRegressionResult"
]