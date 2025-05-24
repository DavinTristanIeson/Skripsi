
import pydantic

class BaseRegressionCoefficient(pydantic.BaseModel):
  name: str
  coefficient: float
  p_value: float
  std_err: float
  sample_size: int
  confidence_interval: tuple[float, float]
  variance_inflation_factor: float

class BaseRegressionIntercept(pydantic.BaseModel):
  intercept: float
  p_value: float
  std_err: float
  sample_size: int
  confidence_interval: tuple[float, float]
  variance_inflation_factor: float