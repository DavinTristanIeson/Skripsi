from enum import Enum
from typing import Optional
import pydantic

from modules.api.enum import ExposedEnum
from modules.table.filter_variants import NamedTableFilter

class LogisticRegressionInterpretation(str, Enum):
  GrandMeanDeviation = "grand_mean_deviation"
  RelativeToReference = "relative_to_reference"
  RelativetoBaseline = "relative_to_baseline"

ExposedEnum().register(LogisticRegressionInterpretation)

class LogisticRegressionInput(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  reference: Optional[str]

class MultinomialLogisticRegressionInput(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  reference_independent: Optional[str]
  reference_dependent: Optional[str]

class LinearRegressionCoefficient(pydantic.BaseModel):
  name: str
  coefficient: float
  p_value: float
  sample_size: int
  z_statistic: float
  confidence_interval: tuple[float, float]
  variance_inflation_factor: float

class LinearRegressionResult(pydantic.BaseModel):
  interpretation: LinearRegressionInterpretation
  coefficients: list[LinearRegressionCoefficient]
  chisq_statistic: float
  p_value: float
  pseudo_r_squared: float
  log_likelihood_ratio: float
  warnings: list[str]
