from typing import Optional
import numpy as np
import pydantic

from modules.regression.results.base import BaseRegressionInput, BaseRegressionResult, RegressionCoefficient, RegressionInterpretation
from modules.table.filter_variants import NamedTableFilter

class LogisticRegressionInput:
  groups: list[NamedTableFilter]
  target: NamedTableFilter
  reference: Optional[str]
  interpretation: RegressionInterpretation
  constrain_by_groups: bool


class MultinomialLogisticRegressionInput(BaseRegressionInput, pydantic.BaseModel):
  reference_dependent: Optional[str]

class LogisticRegressionCoefficient(RegressionCoefficient, pydantic.BaseModel):
  @pydantic.computed_field
  def odds(self)->float:
    return np.exp(self.value)

class LogisticRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  coefficients: list[LogisticRegressionCoefficient]
  intercept: LogisticRegressionCoefficient
  p_value: float
  pseudo_r_squared: float
  log_likelihood_ratio: float

class MultinomialLogisticRegressionFacetResult(pydantic.BaseModel):
  coefficients: list[LogisticRegressionCoefficient]
  intercept: LogisticRegressionCoefficient

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
]