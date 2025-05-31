from typing import Optional
import pydantic

from modules.regression.results.base import BaseRegressionFitEvaluationResult, BaseRegressionInput, BaseRegressionResult, RegressionCoefficient, RegressionPredictionPerIndependentVariableResult

class LinearRegressionInput(BaseRegressionInput, pydantic.BaseModel):
  standardized: bool

class LinearRegressionFitEvaluation(BaseRegressionFitEvaluationResult, pydantic.BaseModel):
  f_statistic: float
  r_squared: float
  rmse: float

class LinearRegressionPredictionResult(pydantic.BaseModel):
  mean: float
  
class LinearRegressionResult(BaseRegressionResult, pydantic.BaseModel):
  coefficients: list[RegressionCoefficient]
  intercept: RegressionCoefficient
  
  # Model fit stats
  fit_evaluation: LinearRegressionFitEvaluation
  predictions: list[LinearRegressionPredictionResult]
  baseline_prediction: LinearRegressionPredictionResult
  standardized: bool


__all__ = [
  "LinearRegressionInput",
  "LinearRegressionResult"
]