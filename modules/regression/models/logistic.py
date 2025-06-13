from dataclasses import dataclass
from typing import cast
import numpy as np
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.regression.models.base import BaseRegressionModel
from modules.regression.models.cache import RegressionModelCacheManager, RegressionModelCacheWrapper
from modules.regression.results.base import RegressionCoefficient, RegressionInterpretation, RegressionPredictionPerIndependentVariableResult
from modules.regression.results.logistic import LogisticRegressionCoefficient, LogisticRegressionFitEvaluation, LogisticRegressionInput, LogisticRegressionPredictionResult, LogisticRegressionResult

def effect_coding_reference_marginal_effect(name: str, margeff: np.ndarray, covariance_matrix: np.ndarray):
  # Effect coding reference effect is the negative sum.
  effect = -margeff.sum()

  # Covariance matrix may contain variances for all independent variables; but to get the variance for the reference variable
  # We need to get the variance of a linear combination (specifically -X1 - X2 - X3 - ... - Xn)
  # We'll use the matrix form rather than the quadratic equation form.
  # https://stats.stackexchange.com/questions/160230/variance-of-linear-combinations-of-correlated-random-variables#:~:text=Or%20with%20a%20matrix%20%28the%20result%20will%20be,A%20on%20the%20off-diagonal%20elements%20in%20the%20result.

  # Assume column vector by default; because math notation.
  weight_vector = np.full((len(margeff), 1), -1)
  # (1, k) x (k, k) x (k, 1) = (1, 1)
  variance_ndarray = weight_vector.T @ covariance_matrix @ weight_vector
  variance = variance_ndarray[0, 0]
  
  # Standard error is just square root of variance (just the std. dev)
  std_err = np.sqrt(variance)
  
  # https://en.wikipedia.org/wiki/Wald_test Wald test for comparing full and null models neatly reduces to effect divided by standard error
  Z = effect / std_err

  import scipy.stats
  # Two-sided p value. Wald test follows the normal distribution.
  p_value = float(2 * (1 - scipy.stats.norm.cdf(abs(Z))))
  alpha = 0.05
  # get the radius from mean.
  ci_radius = scipy.stats.norm.ppf((1 - alpha) / 2)
  
  conf_int = (
    effect - ci_radius * std_err,
    effect + ci_radius * std_err
  )
  return RegressionCoefficient(
    name=name,
    value=effect,
    std_err=std_err,
    statistic=Z,
    p_value=p_value,
    confidence_interval=conf_int,
  )


@dataclass
class LogisticRegressionModel(BaseRegressionModel):
  input: LogisticRegressionInput

  def fit(self):
    # region Preprocess
    input = self.input
    load_result = self._load(
      groups=input.groups,
      target=input.target,
      constrain_by_X=input.constrain_by_groups,
      supported_types=[SchemaColumnTypeEnum.Boolean],
      transform_data=False,
    )
    preprocess_result = self._process_X(
      load_result.X,
      with_intercept=True,
      interpretation=input.interpretation,
      reference=input.reference
    )
    X = preprocess_result.X
    Y = load_result.Y

    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import BinaryResultsWrapper

    # region Fit Model
    Y = Y.astype(np.bool_)
    regression = sm.Logit(Y, X)
    if input.penalty is not None:
      # Model should have supported float. Not sure why the typing is int.
      model = regression.fit_regularized(alpha=cast(int, input.penalty))
    else:
      model = sm.Logit(Y, X).fit(maxiter=100, method="bfgs")

    self.logger.info(model.summary())
    model_id = RegressionModelCacheManager().logistic.save(RegressionModelCacheWrapper(
      model=cast(BinaryResultsWrapper, model),
      levels=None
    ))

    confidence_intervals = model.conf_int()

    results: list[LogisticRegressionCoefficient] = []
    for col_idx, col in enumerate(X.columns):
      results.append(LogisticRegressionCoefficient(
        name=str(col),
        value=model.params[col],
        std_err=model.bse[col],

        statistic=model.tvalues[col],
        p_value=model.pvalues[col],
        confidence_interval=confidence_intervals.loc[col, :],
      ))
    
    # Re-add reference coefficient for GrandMeanDeviation
    remaining_coefficient = self._calculate_remaining_coefficient(
      model=model,
      coefficients=model.params,
      preprocess=preprocess_result,
    )
    if remaining_coefficient is not None and preprocess_result.reference_idx is not None:
      # Make sure to add the odds_ratio and odds_ratio_confidence_interval via LogisticRegressionCoefficient
      results.insert(preprocess_result.reference_idx, LogisticRegressionCoefficient.model_validate(
        remaining_coefficient,
        from_attributes=True,
      ))

    results[0].name = "Baseline"


    marginal_effects: list[RegressionCoefficient] = []
    # Use AME
    raw_marginal_effects = model.get_margeff(at="overall", method="dydx", dummy=True)
    self.logger.info(raw_marginal_effects.summary())
    raw_marginal_effects_conf_int = raw_marginal_effects.conf_int()
    # Intercept is excluded. Skip the first column.
    for col_idx, col in enumerate(X.columns[1:]):
      marginal_effects.append(RegressionCoefficient(
        name=str(col),
        value=raw_marginal_effects.margeff[col_idx],
        std_err=raw_marginal_effects.margeff_se[col_idx],
        statistic=raw_marginal_effects.tvalues[col_idx],
        p_value=raw_marginal_effects.pvalues[col_idx],
        confidence_interval=raw_marginal_effects_conf_int[col_idx],
      ))
    
    # Re-add reference coefficient for GrandMeanDeviation
    if (
      input.interpretation == RegressionInterpretation.GrandMeanDeviation and 
      preprocess_result.reference is not None and
      preprocess_result.reference_idx is not None
    ):
      remaining_marginal_effect = effect_coding_reference_marginal_effect(
        name=str(preprocess_result.reference.name),
        margeff=raw_marginal_effects.margeff,
        covariance_matrix=raw_marginal_effects.margeff_cov,
      )
      marginal_effects.insert(preprocess_result.reference_idx, remaining_marginal_effect)

    # region Predictions
    model_predictions = model.predict(
      self._regression_prediction_input(preprocess_result)
    )

    prediction_results = list(map(
      lambda coefficient, probability: RegressionPredictionPerIndependentVariableResult(
        variable=coefficient.name,
        prediction=LogisticRegressionPredictionResult(
          probability=probability,
        )
      ),
      results,
      model_predictions
    ))

    result = LogisticRegressionResult(
      model_id=model_id,
      reference=preprocess_result.reference_name,
      independent_variables=preprocess_result.independent_variables,
      interpretation=input.interpretation,

      intercept = results[0],
      coefficients = results[1:],
      marginal_effects=marginal_effects,

      sample_size=len(Y),
      warnings=[],

      fit_evaluation=LogisticRegressionFitEvaluation(
        converged=model.mle_retvals.get('converged', True),
        p_value=model.llr_pvalue,
        model_dof=model.df_model,
        residual_dof=model.df_resid,
        pseudo_r_squared=model.prsquared,
        log_likelihood_ratio=model.llr,
        log_likelihood=model.llf,
        log_likelihood_null=model.llnull,
        aic=model.aic,
        bic=model.bic,
      ),
      predictions=prediction_results[1:],
      baseline_prediction=prediction_results[0].prediction,
    )
    return result
