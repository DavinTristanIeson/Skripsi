import numpy as np
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.regression.models.base import BaseRegressionModel
from modules.regression.results.base import RegressionCoefficient
from modules.regression.results.linear import LinearRegressionInput, LinearRegressionResult

class LinearRegressionModel(BaseRegressionModel):
  input: LinearRegressionInput
  def fit(self):
    input = self.input
    X, Y = self._load(
      groups=input.groups,
      target=input.target,
      constrain_by_X=input.constrain_by_groups,
      supported_types=[SchemaColumnTypeEnum.Continuous]
    )
    X = self._process_X(
      X,
      with_intercept=True,
      interpretation=input.interpretation,
      reference=input.reference
    )

    from sklearn.discriminant_analysis import StandardScaler
    if input.standardized:
      Y = StandardScaler().fit_transform(Y.to_numpy().reshape(-1, 1))[0, :]

    warnings = []
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    model = sm.OLS(Y, X).fit()
    results: list[RegressionCoefficient] = []
    confidence_intervals = model.conf_int()
    for col_idx, col in enumerate(X.columns):
      results.append(RegressionCoefficient(
        name=str(col),

        value = model.params[col],
        std_err = model.bse[col],
        sample_size=X[col].sum(),

        statistic = model.tvalues[col],
        p_value = model.pvalues[col],
        confidence_interval=confidence_intervals[col],

        variance_inflation_factor=variance_inflation_factor(X, col_idx),
      ))
    
    return LinearRegressionResult(
      reference=input.reference,
      converged=True, # OLS has no convergence concept
      coefficients=results[1:],
      intercept=results[0],
      f_statistic=model.fvalue,
      p_value=model.f_pvalue,
      r_squared=model.rsquared,
      standardized=input.standardized,
      interpretation=input.interpretation,
      sample_size=len(Y),
      warnings=warnings,
      rmse=np.sqrt(model.mse_resid)
    )
