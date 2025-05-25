from modules.regression.models.base import BaseRegressionModel
from modules.regression.results.linear import LinearRegressionInput


class LinearRegressionModel(BaseRegressionModel):
  input: LinearRegressionInput
  def fit(self):
    X, Y = self._load()
    import statsmodels.api as sm

    sm.OLS(Y, X)