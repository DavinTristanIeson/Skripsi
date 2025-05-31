import numpy as np
import pydantic

from modules.regression.results.base import BaseRegressionFitEvaluationResult, BaseRegressionInput, BaseRegressionResult, OddsBasedRegressionCoefficient, RegressionCoefficient, RegressionDependentVariableLevelInfo, RegressionPredictionPerIndependentVariableResult

class OrdinalRegressionInput(BaseRegressionInput, pydantic.BaseModel):
  pass

class OrdinalRegressionThreshold(pydantic.BaseModel):
  from_level: str
  to_level: str
  value: float
  @pydantic.computed_field
  def odds_ratio(self)->float:
    return np.exp(self.value)

class OrdinalRegressionCoefficient(OddsBasedRegressionCoefficient, pydantic.BaseModel):
  pass

class OrdinalRegressionFitEvaluation(BaseRegressionFitEvaluationResult, pydantic.BaseModel):
  log_likelihood_ratio: float
  pseudo_r_squared: float

class OrdinalRegressionPredictionResult(pydantic.BaseModel):
  latent_score: float
  probabilities: list[float]
  levels: list[str]

class OrdinalRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  coefficients: list[OrdinalRegressionCoefficient]
  thresholds: list[OrdinalRegressionThreshold]
  levels: list[RegressionDependentVariableLevelInfo]
  fit_evaluation: OrdinalRegressionFitEvaluation
  predictions: list[OrdinalRegressionPredictionResult]
  baseline_prediction: OrdinalRegressionPredictionResult

__all__ = [
  "OrdinalRegressionResult"
]