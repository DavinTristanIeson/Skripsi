from dataclasses import dataclass
from typing import Any, Optional, cast
import numpy as np
import pandas as pd
from modules.config.schema.base import CATEGORICAL_SCHEMA_COLUMN_TYPES, SchemaColumnTypeEnum
from modules.regression.exceptions import DependentVariableReferenceMustBeAValidValueException, MultilevelRegressionNotEnoughLevelsException
from modules.regression.models.base import BaseRegressionModel, RegressionProcessXResult
from modules.regression.models.cache import RegressionModelCacheManager, RegressionModelCacheWrapper
from modules.regression.models.logistic import effect_coding_reference_marginal_effect
from modules.regression.results.base import RegressionCoefficient, RegressionInterpretation, RegressionDependentVariableLevelInfo, RegressionPredictionPerIndependentVariableResult
from modules.regression.results.logistic import LogisticRegressionCoefficient, LogisticRegressionFitEvaluation, LogisticRegressionInput, LogisticRegressionPredictionResult, MultinomialLogisticRegressionFacetResult, MultinomialLogisticRegressionMarginalEffectsFacetResult, MultinomialLogisticRegressionPredictionResult, MultinomialLogisticRegressionResult, LogisticRegressionResult, MultinomialLogisticRegressionInput


@dataclass
class MultinomialLogisticRegressionModel(BaseRegressionModel):
  input: MultinomialLogisticRegressionInput

  def _calculate_remaining_coefficient_logistic(
    self,
    model: Any,
    coefficients: pd.Series,
    preprocess: RegressionProcessXResult,
    interpretation: RegressionInterpretation,
    idx: int,
  )->Optional[RegressionCoefficient]:
    if interpretation != RegressionInterpretation.GrandMeanDeviation or preprocess.reference is None:
      return None
    reference_coefficient = -coefficients.iloc[1:, idx].sum() # type: ignore
    # T-test accepts shape of (features, classes - 1) that is flattened; but the params from the model is (classes - 1, features)
    # That's why coefficient has to be transposed.
    coefficient_weights = np.full(coefficients.T.shape, 0)
    # Set the idx (corresponds to the class) and 1: (corresponds to the coefficients, 0 is intercept) to -1.
    # This means that we only test the effect coding remaining coefficient.
    coefficient_weights[idx, 1:] = -1

    # It may be called a T-test, but the computation is based on the underlying model's distribution
    # Verify that you passed the correct order of coefficients by using the reference_coefficient as sanity check
    test_result = model.t_test(coefficient_weights.flat)

    return RegressionCoefficient(
      name=str(preprocess.reference.name),
      value=test_result.effect[0],
      p_value=test_result.pvalue,
      std_err=test_result.sd[0, 0],
      confidence_interval=test_result.conf_int()[0].tolist(),
      statistic=test_result.statistic[0, 0],
    )
  

  def fit(self):
    # region Preprocess
    input = self.input
    load_result = self._load(
      groups=input.groups,
      target=input.target,
      constrain_by_X=input.constrain_by_groups,
      supported_types=list(filter(
        lambda type: type != SchemaColumnTypeEnum.Boolean,
        CATEGORICAL_SCHEMA_COLUMN_TYPES
      ))
    )
    preprocess_result = self._process_X(
      load_result.X,
      with_intercept=True,
      interpretation=input.interpretation,
      reference=input.reference
    )
    X = preprocess_result.X
    Y = load_result.Y

    # Make sure that the reference dependent variable is chosen by statsmodels (urgh, why can't we manually specify; maybe it's a Patsy option?)
    cat_Y = pd.Categorical(Y)
    cat_Y = cat_Y.remove_unused_categories()
    
    reference_dependent = input.reference_dependent or cat_Y.categories[0]
    if reference_dependent not in cat_Y.categories:
      raise DependentVariableReferenceMustBeAValidValueException(reference=reference_dependent, supported_values=cat_Y.categories)
    Y_categories = list(cat_Y.categories)
    idx = Y_categories.index(input.reference_dependent)
    if idx == -1:
      raise ValueError("Can't find the category that corresponds to reference dependent. This might be a developer oversight.")
    # Move the category to the front
    category = Y_categories.pop(idx)
    Y_categories.insert(0, category)
    # Make the reference category first so that the model automatically uses it as the baseline.
    cat_Y = cat_Y.reorder_categories(Y_categories)
    cat_Y = cat_Y.rename_categories({
      # Ensures that this is first alphabetically
      # The category reorder is for our own purposes (Y.cat.categories is important, so make sure that reference is in front!), the renaming is for statsmodels.
      input.reference_dependent: "",
    })

    Y = pd.Series(cat_Y, index=Y.index)
    MultilevelRegressionNotEnoughLevelsException.assert_levels("Multinomial Logistic", Y_categories)

    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import MultinomialResultsWrapper
    # region Fit Model
    # Newton solver produces NaN too often.
    regression = sm.MNLogit(Y, X)
    if input.penalty is not None:
      # Model should have supported float. Not sure why the typing is int.
      model = regression.fit_regularized(alpha=cast(int, input.penalty))
    else:
      model = regression.fit(maxiter=300, method="bfgs")
  
    self.logger.info(model.summary())

    # Get dependent variable levels
    dependent_variable_levels: list[RegressionDependentVariableLevelInfo] = self._dependent_variable_levels(
      Y,
      reference_dependent=reference_dependent,
      column=load_result.column
    )
    model_id = RegressionModelCacheManager().multinomial_logistic.save(RegressionModelCacheWrapper(
      model=cast(MultinomialResultsWrapper, model),
      # Store the levels since statsmodels doesn't.
      levels=list(map(lambda level: level.name, dependent_variable_levels))
    ))

    if input.penalty is not None:
      confidence_intervals = model._results.conf_int()
    else:
      confidence_intervals = model.conf_int()

    facets: list[MultinomialLogisticRegressionFacetResult] = []

    # First category is excluded. First category is guaranteed to be the reference_dependent from the code above.
    for row_idx, row in enumerate(Y.cat.categories[1:]):
      results: list[LogisticRegressionCoefficient] = []
      for col_idx, col in enumerate(X.columns):
        results.append(LogisticRegressionCoefficient(
          name=str(col),

          value=model.params.at[col, row_idx],
          std_err=model.bse.at[col, row_idx],

          statistic=model.tvalues.at[col, row_idx],
          p_value=model.pvalues.at[col, row_idx],
          confidence_interval=(
            confidence_intervals.loc[(row, col)].to_list()
            if hasattr(confidence_intervals, "loc")
            else confidence_intervals[row_idx, col_idx].tolist()
          ),
        ))

      # Add the reference coefficient (this has to be done for every MNLogit level)
      remaining_coefficient = self._calculate_remaining_coefficient_logistic(
        model=model,
        coefficients=model.params,
        interpretation=input.interpretation,
        preprocess=preprocess_result,
        idx=row_idx,
      )
      if remaining_coefficient is not None and preprocess_result.reference_idx is not None:
        results.insert(preprocess_result.reference_idx, LogisticRegressionCoefficient.model_validate(
          remaining_coefficient,
          from_attributes=True,
        ))

      # First coefficient is always intercept.
      intercept = results[0]
      intercept.name = "Baseline"
      facets.append(MultinomialLogisticRegressionFacetResult(
        level=row,
        # Coefficients is everything after intercept
        coefficients=results[1:],
        intercept=intercept,
      ))

    # Use AME
    # MNLogit has a bug with marginal effects wherein it doesn't support dummy=True.
    # Using dummy=False is technically incorrect for binary variables since something like X=0.1 doesn't make sense, but it's a good enough approximation
    raw_marginal_effects = model.get_margeff(at="overall", method="dydx", dummy=False)
    self.logger.info(raw_marginal_effects.summary())
    raw_marginal_effects_conf_int = raw_marginal_effects.conf_int()
    marginal_effects: list[MultinomialLogisticRegressionMarginalEffectsFacetResult] = []
    # First category is included
    for row_idx, category in enumerate(Y_categories):
      marginal_effects_row: list[RegressionCoefficient] = []
      # but skip intercept
      for col_idx, column in enumerate(X.columns[1:]):
        marginal_effects_row.append(RegressionCoefficient(
          name=column,
          value=raw_marginal_effects.margeff[col_idx, row_idx],
          std_err=raw_marginal_effects.margeff_se[col_idx, row_idx],
          statistic=raw_marginal_effects.tvalues[col_idx, row_idx],
          p_value=raw_marginal_effects.pvalues[col_idx, row_idx],
          confidence_interval=(
            raw_marginal_effects_conf_int[col_idx, 0, row_idx],
            raw_marginal_effects_conf_int[col_idx, 1, row_idx],
          ),
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
        marginal_effects_row.insert(preprocess_result.reference_idx, remaining_marginal_effect)

      marginal_effects.append(MultinomialLogisticRegressionMarginalEffectsFacetResult(
        level=category,
        marginal_effects=marginal_effects_row,
      ))

    # region Prediction
    raw_levels = list(map(lambda level: level.name, dependent_variable_levels))
    model_predictions = model.predict(
      self._regression_prediction_input(preprocess_result)
    )
    prediction_results = list(map(
      lambda coefficient, result: RegressionPredictionPerIndependentVariableResult(
        variable=coefficient.name,
        prediction=MultinomialLogisticRegressionPredictionResult(
          probabilities=result,
          levels=raw_levels,
        )
      ),
      [facets[0].intercept, *facets[0].coefficients],
      model_predictions,
    ))

  
    result = MultinomialLogisticRegressionResult(
      model_id=model_id,
      reference=preprocess_result.reference_name,
      reference_dependent=reference_dependent,
      levels=dependent_variable_levels,
      independent_variables=preprocess_result.independent_variables,
      interpretation=input.interpretation,

      facets=facets,
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
