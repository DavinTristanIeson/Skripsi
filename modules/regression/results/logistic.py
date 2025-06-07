from typing import Optional
import pydantic

from modules.regression.results.base import BaseRegressionFitEvaluationResult, BaseRegressionInput, BaseRegressionResult, OddsBasedRegressionCoefficient, RegressionCoefficient, RegressionInterpretation, RegressionDependentVariableLevelInfo, RegressionPredictionPerIndependentVariableResult
from modules.table.filter_variants import NamedTableFilter

class LogisticRegressionInput(BaseRegressionInput, pydantic.BaseModel):
  target: NamedTableFilter
  reference: Optional[str]
  interpretation: RegressionInterpretation
  constrain_by_groups: bool


class MultinomialLogisticRegressionInput(BaseRegressionInput, pydantic.BaseModel):
  reference_dependent: Optional[str]
  target: list[NamedTableFilter] | str

class LogisticRegressionCoefficient(OddsBasedRegressionCoefficient, pydantic.BaseModel):
  pass

class LogisticRegressionPredictionResult(pydantic.BaseModel):
  probability: float
class LogisticRegressionFitEvaluation(BaseRegressionFitEvaluationResult):
  pseudo_r_squared: float
  log_likelihood_ratio: float

class LogisticRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  coefficients: list[LogisticRegressionCoefficient]
  intercept: LogisticRegressionCoefficient
  fit_evaluation: LogisticRegressionFitEvaluation
  predictions: list[RegressionPredictionPerIndependentVariableResult[LogisticRegressionPredictionResult]]
  baseline_prediction: LogisticRegressionPredictionResult

class MultinomialLogisticRegressionFacetResult(pydantic.BaseModel):
  level: str
  coefficients: list[LogisticRegressionCoefficient]
  intercept: LogisticRegressionCoefficient

class MultinomialLogisticRegressionPredictionResult(pydantic.BaseModel):
  probabilities: list[float]
  levels: list[str]
  
class MultinomialLogisticRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  reference: Optional[str]
  reference_dependent: str
  levels: list[RegressionDependentVariableLevelInfo]

  facets: list[MultinomialLogisticRegressionFacetResult]
  fit_evaluation: LogisticRegressionFitEvaluation
  predictions: list[RegressionPredictionPerIndependentVariableResult[MultinomialLogisticRegressionPredictionResult]]
  baseline_prediction: MultinomialLogisticRegressionPredictionResult


__all__ = [
  "MultinomialLogisticRegressionInput",
  "LogisticRegressionResult",
  "MultinomialLogisticRegressionFacetResult",
  "MultinomialLogisticRegressionResult",
]