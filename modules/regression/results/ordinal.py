import numpy as np
import pydantic

from modules.regression.results.base import BaseRegressionFitEvaluationResult, BaseRegressionInput, BaseRegressionResult, LogLikelihoodBasedFitEvaluation, OddsBasedRegressionCoefficient, RegressionCoefficient, RegressionDependentVariableLevelInfo, RegressionPredictionPerIndependentVariableResult
from modules.table.filter_variants import NamedTableFilter

class OrdinalRegressionInput(BaseRegressionInput, pydantic.BaseModel):
  target: list[NamedTableFilter] | str

class OrdinalRegressionThreshold(pydantic.BaseModel):
  from_level: str
  to_level: str
  value: float

class OrdinalRegressionCoefficient(OddsBasedRegressionCoefficient, pydantic.BaseModel):
  pass

class OrdinalRegressionFitEvaluation(LogLikelihoodBasedFitEvaluation, pydantic.BaseModel):
  pass

class OrdinalRegressionPredictionResult(pydantic.BaseModel):
  latent_score: float
  probabilities: list[float]
  cumulative_probabilities: list[float]
  levels: list[str]

class OrdinalRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  coefficients: list[OrdinalRegressionCoefficient]
  thresholds: list[OrdinalRegressionThreshold]
  levels: list[RegressionDependentVariableLevelInfo]
  fit_evaluation: OrdinalRegressionFitEvaluation
  predictions: list[RegressionPredictionPerIndependentVariableResult[OrdinalRegressionPredictionResult]]
  baseline_prediction: OrdinalRegressionPredictionResult

__all__ = [
  "OrdinalRegressionResult"
]