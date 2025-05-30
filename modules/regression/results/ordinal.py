import numpy as np
import pydantic

from modules.regression.results.base import BaseRegressionInput, BaseRegressionResult, OddsBasedRegressionCoefficient, RegressionCoefficient

class OrdinalRegressionInput(BaseRegressionInput, pydantic.BaseModel):
  pass

class OrdinalRegressionThreshold(pydantic.BaseModel):
  from_level: str
  to_level: str
  value: float
  @pydantic.computed_field
  def odds_ratio(self)->float:
    return np.exp(self.value)

class OrdinalRegressionLevelSampleSize(pydantic.BaseModel):
  name: str
  sample_size: int

class OrdinalRegressionCoefficient(OddsBasedRegressionCoefficient, pydantic.BaseModel):
  pass

class OrdinalRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  coefficients: list[OrdinalRegressionCoefficient]
  thresholds: list[OrdinalRegressionThreshold]
  sample_sizes: list[OrdinalRegressionLevelSampleSize]
  
  log_likelihood_ratio: float
  p_value: float
  pseudo_r_squared: float
  converged: bool
  warnings: list[str]

__all__ = [
  "OrdinalRegressionResult"
]