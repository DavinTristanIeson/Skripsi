import numpy as np
from modules.project.cache import ProjectCache
from modules.regression.models.cache import RegressionModelCacheManager
from modules.regression.models.linear import LinearRegressionModel
from modules.regression.models.logistic import LogisticRegressionModel, MultinomialLogisticRegressionModel
from modules.regression.models.ordinal import OrdinalRegressionModel
from modules.regression.results.base import BaseRegressionPredictionInput
from modules.regression.results.linear import LinearRegressionInput, LinearRegressionPredictionResult, LinearRegressionResult
from modules.regression.results.logistic import LogisticRegressionInput, LogisticRegressionPredictionResult, LogisticRegressionResult, MultinomialLogisticRegressionInput, MultinomialLogisticRegressionPredictionResult, MultinomialLogisticRegressionResult
from modules.regression.results.ordinal import OrdinalRegressionInput, OrdinalRegressionPredictionResult, OrdinalRegressionResult

def linear_regression(cache: ProjectCache, input: LinearRegressionInput)->LinearRegressionResult:
  return LinearRegressionModel(cache=cache, input=input).fit()

def logistic_regression(cache: ProjectCache, input: LogisticRegressionInput)->LogisticRegressionResult:
  return LogisticRegressionModel(cache=cache, input=input).fit()

def multinomial_logistic_regression(cache: ProjectCache, input: MultinomialLogisticRegressionInput)->MultinomialLogisticRegressionResult:
  return MultinomialLogisticRegressionModel(cache=cache, input=input).fit()

def ordinal_regression(cache: ProjectCache, input: OrdinalRegressionInput)->OrdinalRegressionResult:
  return OrdinalRegressionModel(cache=cache, input=input).fit()

def linear_regression_predict(input: BaseRegressionPredictionInput):
  model = RegressionModelCacheManager().linear.load(input.model_id)
  X = input.as_regression_input()
  Y = model.predict(X)
  return LinearRegressionPredictionResult(
    mean=Y[0],
  )

def logistic_regression_predict(input: BaseRegressionPredictionInput):
  model = RegressionModelCacheManager().logistic.load(input.model_id)
  X = input.as_regression_input()
  Y = model.predict(X)
  return LogisticRegressionPredictionResult(
    probability=Y[0]
  )

def multinomial_logistic_regression_predict(input: BaseRegressionPredictionInput):
  model = RegressionModelCacheManager().multinomial_logistic.load(input.model_id)
  X = input.as_regression_input()
  Y = model.predict(X)
  return MultinomialLogisticRegressionPredictionResult(
    probabilities=Y[0],
    levels=Y.columns,
  )

def ordinal_regression_predict(input: BaseRegressionPredictionInput):
  model = RegressionModelCacheManager().ordinal.load(input.model_id)
  X = np.array([[True, *input.input]])
  latent_variable = model.predict(X, which="linear")[0]
  probabilities = model.predict(X, which="prob")[0]
  return OrdinalRegressionPredictionResult(
    probabilities=probabilities, # type: ignore
    latent_variable=latent_variable, # type: ignore
    levels=probabilities.columns,
  )

__all__ = [
  "linear_regression",
  "logistic_regression",
  "multinomial_logistic_regression",
  "ordinal_regression",
  "linear_regression_predict",
  "logistic_regression_predict",
  "multinomial_logistic_regression_predict",
  "ordinal_regression_predict"
]