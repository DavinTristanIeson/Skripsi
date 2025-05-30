from dataclasses import dataclass
from typing import cast
import numpy as np
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.logger.provisioner import ProvisionedLogger
from modules.regression.models.base import BaseRegressionModel
from modules.regression.models.cache import RegressionModelCacheManager
from modules.regression.results.base import RegressionCoefficient, RegressionPredictionPerIndependentVariableResult
from modules.regression.results.linear import LinearRegressionFitEvaluation, LinearRegressionInput, LinearRegressionPredictionResult, LinearRegressionResult


@dataclass
class LinearRegressionModel(BaseRegressionModel):
  input: LinearRegressionInput
  def fit(self):
    input = self.input
    load_result = self._load(
      groups=input.groups,
      target=input.target,
      constrain_by_X=input.constrain_by_groups,
      supported_types=[SchemaColumnTypeEnum.Continuous]
    )
    preprocess_result = self._process_X(
      load_result.X,
      with_intercept=True,
      interpretation=input.interpretation,
      reference=input.reference
    )
    X = preprocess_result.X
    Y = load_result.Y

    from sklearn.discriminant_analysis import StandardScaler
    if input.standardized:
      Y = StandardScaler().fit_transform(Y.to_numpy().reshape(-1, 1))[:, 0]

    warnings = []
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    model = sm.OLS(Y, X).fit()
    self.logger.info(model.summary())
    model_id = RegressionModelCacheManager().linear.save(model) # type: ignore

    results: list[RegressionCoefficient] = []
    confidence_intervals = model.conf_int()
    for col_idx, col in enumerate(X.columns):
      results.append(RegressionCoefficient(
        name=str(col),

        value = model.params[col],
        std_err = model.bse[col],
        sample_size=preprocess_result.sample_sizes[col],

        statistic = model.tvalues[col],
        p_value = model.pvalues[col],
        confidence_interval=confidence_intervals.loc[col, :],

        variance_inflation_factor=variance_inflation_factor(X, col_idx),
      ))

    remaining_coefficient = self._calculate_remaining_coefficient(
      model=model,
      coefficients=model.params,
      interpretation=input.interpretation,
      preprocess=preprocess_result,
    )
    if remaining_coefficient is not None:
      results.append(remaining_coefficient)

    model_predictions = model.predict(
      self._regression_prediction_input(load_result.independent_variables)
    )

    prediction_results = list(map(
      lambda result: LinearRegressionPredictionResult(
        mean=result,
      ),
      model_predictions
    ))
    
    return LinearRegressionResult(
      model_id=model_id,
      independent_variables=load_result.independent_variables,
      reference=preprocess_result.reference_name,

      coefficients=results[1:],
      intercept=results[0],
      sample_size=len(Y),

      interpretation=input.interpretation,
      standardized=input.standardized,

      fit_evaluation=LinearRegressionFitEvaluation(
        converged=True, # OLS has no convergence concept
        f_statistic=model.fvalue,
        p_value=model.f_pvalue,
        r_squared=model.rsquared_adj,
        rmse=np.sqrt(model.mse_resid)
      ),
      predictions=prediction_results[1:],
      baseline_prediction=prediction_results[0],

      warnings=warnings,
    )
