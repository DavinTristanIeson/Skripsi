from typing import Optional
import pydantic

from modules.regression.results.base import BaseRegressionInput, BaseRegressionResult, RegressionCoefficient

class LinearRegressionInput(BaseRegressionInput, pydantic.BaseModel):
  standardized: bool

class LinearRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  coefficients: list[RegressionCoefficient]
  intercept: Optional[RegressionCoefficient]
  f_statistic: float
  p_value: float
  r_squared: float
  standardized: bool
  rmse: float

class LinearRegressionPredictionResult(pydantic.BaseModel):
  mean: float
  

__all__ = [
  "LinearRegressionInput",
  "LinearRegressionResult"
]