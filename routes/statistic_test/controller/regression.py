from modules.config.schema.base import SchemaColumnTypeEnum
from modules.project.cache import ProjectCache
from modules.regression.models.linear import LinearRegressionModel
from modules.regression.models.logistic import LogisticRegressionModel, MultinomialLogisticRegressionModel, OneVsRestLogisticRegressionModel
from modules.regression.models.ordinal import OrdinalRegressionModel
from modules.regression.results.base import BaseRegressionInput
from modules.regression.results.linear import LinearRegressionInput, LinearRegressionResult
from modules.regression.results.logistic import LogisticRegressionResult, MultinomialLogisticRegressionInput, MultinomialLogisticRegressionResult, OneVsRestLogisticRegressionResult
from modules.regression.results.ordinal import OrdinalRegressionResult


def linear_regression(cache: ProjectCache, input: LinearRegressionInput)->LinearRegressionResult:
  return LinearRegressionModel(cache=cache, input=input).fit()

def logistic_regression(cache: ProjectCache, input: BaseRegressionInput)->LogisticRegressionResult:
  return LogisticRegressionModel(cache=cache, input=input).fit()

def one_vs_rest_logistic_regression(cache: ProjectCache, input: BaseRegressionInput)->OneVsRestLogisticRegressionResult:
  return OneVsRestLogisticRegressionModel(cache=cache, input=input).fit()

def multinomial_logistic_regression(cache: ProjectCache, input: MultinomialLogisticRegressionInput)->MultinomialLogisticRegressionResult:
  return MultinomialLogisticRegressionModel(cache=cache, input=input).fit()

def ordinal_regression(cache: ProjectCache, input: BaseRegressionInput)->OrdinalRegressionResult:
  return OrdinalRegressionModel(cache=cache, input=input).fit()

__all__ = [
  "linear_regression",
  "logistic_regression",
  "one_vs_rest_logistic_regression",
  "multinomial_logistic_regression",
  "ordinal_regression",
]