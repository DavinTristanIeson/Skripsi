from enum import Enum
from typing import Optional
import pydantic

from modules.api.enum import ExposedEnum
from modules.regression.results.base import BaseRegressionCoefficient
from modules.table.filter_variants import NamedTableFilter

class LinearRegressionInterpretation(str, Enum):
  GrandMeanDeviation = "grand_mean_deviation"
  RelativeToReference = "relative_to_reference"
  RelativetoBaseline = "relative_to_baseline"

ExposedEnum().register(LinearRegressionInterpretation)

class LinearRegressionInput(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  reference: Optional[str]
  with_intercept: bool
  standardized: bool

class LinearRegressionCoefficient(BaseRegressionCoefficient, pydantic.BaseModel):
  t_statistic: float

class LinearRegressionIntercept:
  

class LinearRegressionResult(pydantic.BaseModel):
  interpretation: LinearRegressionInterpretation
  coefficients: list[LinearRegressionCoefficient]
  intercept: Optional[float]
  f_statistic: float
  p_value: float
  r_squared: float
  warnings: list[str]
