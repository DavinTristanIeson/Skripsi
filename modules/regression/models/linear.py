from dataclasses import dataclass
from typing import cast
import numpy as np
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.logger.provisioner import ProvisionedLogger
from modules.regression.exceptions import RegressionFailedException
from modules.regression.models.base import BaseRegressionModel
from modules.regression.models.cache import RegressionModelCacheManager, RegressionModelCacheWrapper
from modules.regression.results.base import RegressionCoefficient, RegressionPredictionPerIndependentVariableResult
from modules.regression.results.linear import LinearRegressionFitEvaluation, LinearRegressionInput, LinearRegressionPredictionResult, LinearRegressionResult


@dataclass
class LinearRegressionModel(BaseRegressionModel):
  input: LinearRegressionInput
  def fit(self):
    # region Preprocess
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
    # standardize Y if requested by user.
    if input.standardized:
      Y = StandardScaler().fit_transform(Y.to_numpy().reshape(-1, 1))[:, 0]

    import statsmodels.api as sm

    # region Fitting
    try:
      model = sm.OLS(Y, X).fit(cov_type="HC3")
      self.logger.info(model.summary())
    except Exception as e:
      raise RegressionFailedException(e)

    model_id = RegressionModelCacheManager().linear.save(RegressionModelCacheWrapper(
      model=model,
      levels=None
    ))

    results: list[RegressionCoefficient] = []
    confidence_intervals = model.conf_int()
    # Coefficients is flat.
    for col_idx, col in enumerate(X.columns):
      results.append(RegressionCoefficient(
        name=str(col),

        value = model.params[col],
        std_err = model.bse[col],

        statistic = model.tvalues[col],
        p_value = model.pvalues[col],
        confidence_interval=confidence_intervals.loc[col, :],
      ))

    # Re-add missing coefficient for GrandMeanDeviation
    remaining_coefficient = self._calculate_remaining_coefficient(
      model=model,
      coefficients=model.params,
      preprocess=preprocess_result,
    )
    if remaining_coefficient is not None and preprocess_result.reference_idx is not None:
      results.insert(preprocess_result.reference_idx, remaining_coefficient)
    results[0].name = "Baseline"

    # region Predictions
    model_predictions = model.predict(
      self._regression_prediction_input(preprocess_result)
    )

    prediction_results = list(map(
      lambda coefficient, result: RegressionPredictionPerIndependentVariableResult(
        variable=coefficient.name,
        prediction=LinearRegressionPredictionResult(
          mean=result,
        )
      ),
      results,
      model_predictions,
    ))
    
    result = LinearRegressionResult(
      model_id=model_id,
      independent_variables=preprocess_result.independent_variables,
      reference=preprocess_result.reference_name,

      coefficients=results[1:],
      intercept=results[0],
      sample_size=len(Y),

      interpretation=input.interpretation,
      standardized=input.standardized,

      fit_evaluation=LinearRegressionFitEvaluation(
        converged=True, # OLS has no convergence concept
        f_statistic=model.fvalue,
        model_dof=model.df_model,
        residual_dof=model.df_resid,
        p_value=model.f_pvalue,
        r_squared=model.rsquared_adj,
        rmse=np.sqrt(model.mse_resid),
        log_likelihood=model.llf,
        aic=model.aic,
        bic=model.bic,
      ),
      predictions=prediction_results[1:],
      baseline_prediction=prediction_results[0].prediction,

      warnings=[],
    )
    return result
