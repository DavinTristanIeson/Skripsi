from enum import Enum
from typing import Optional
import pydantic

class RegressionInterpretation(str, Enum):
  pass

class RegressionCoefficient(pydantic.BaseModel):
  name: str
  coefficient: float
  p_value: float
  std_err: float
  sample_size: int
  confidence_interval: tuple[float, float]
  variance_inflation_factor: float
  statistic: float

class RegressionIntercept(pydantic.BaseModel):
  intercept: float
  p_value: float
  std_err: float
  sample_size: int
  confidence_interval: tuple[float, float]
  variance_inflation_factor: float

class BaseRegressionResult(pydantic.BaseModel):
  reference: Optional[str]
  converged: bool
  sample_size: int
  warnings: list[str]