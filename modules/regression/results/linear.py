import pydantic

from modules.regression.results.base import BaseRegressionFitEvaluationResult, BaseRegressionInput, BaseRegressionResult, RegressionCoefficient, RegressionPredictionPerIndependentVariableResult

class LinearRegressionInput(BaseRegressionInput, pydantic.BaseModel):
  target: str
  standardized: bool

class LinearRegressionFitEvaluation(BaseRegressionFitEvaluationResult, pydantic.BaseModel):
  f_statistic: float
  r_squared: float
  rmse: float
  log_likelihood: float
  aic: float
  bic: float

class LinearRegressionPredictionResult(pydantic.BaseModel):
  mean: float
  
class LinearRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  coefficients: list[RegressionCoefficient]
  intercept: RegressionCoefficient
  
  # Model fit stats
  fit_evaluation: LinearRegressionFitEvaluation
  predictions: list[RegressionPredictionPerIndependentVariableResult[LinearRegressionPredictionResult]]
  baseline_prediction: LinearRegressionPredictionResult
  standardized: bool


__all__ = [
  "LinearRegressionInput",
  "LinearRegressionResult"
]