from dataclasses import dataclass
from typing import Any, Sequence, cast
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

from modules.config.schema.base import ORDERED_CATEGORICAL_SCHEMA_COLUMN_TYPES
from modules.regression.exceptions import OrdinalRegressionNotEnoughLevelsException
from modules.regression.models.base import BaseRegressionModel
from modules.regression.models.cache import RegressionModelCacheManager, RegressionModelCacheWrapper
from modules.regression.results.base import RegressionDependentVariableLevelInfo, RegressionPredictionPerIndependentVariableResult
from modules.regression.results.ordinal import OrdinalRegressionCoefficient, OrdinalRegressionFitEvaluation, OrdinalRegressionInput, OrdinalRegressionPredictionResult, OrdinalRegressionThreshold, OrdinalRegressionResult

@dataclass
class OrdinalRegressionModel(BaseRegressionModel):
  input: OrdinalRegressionInput

  def fit(self):
    # region Preprocess
    input = self.input

    load_result = self._load(
      groups=input.groups,
      target=input.target,
      constrain_by_X=input.constrain_by_groups,
      supported_types=ORDERED_CATEGORICAL_SCHEMA_COLUMN_TYPES
    )
    preprocess_result = self._process_X(
      load_result.X,
      with_intercept=False,
      interpretation=input.interpretation,
      reference=input.reference
    )
    X = preprocess_result.X
    Y = load_result.Y

    warnings = []

    cat_Y_categories = pd.Categorical(Y.sort_values(), ordered=True).categories
    cat_Y = pd.Categorical(Y, categories=cat_Y_categories, ordered=True)
    Y = pd.Series(cat_Y, index=Y.index)

    # Don't bother with unused categories. 
    Y = Y.cat.remove_unused_categories()
    levels = Y.cat.categories
    # Make sure there's more than 2 levels.
    OrdinalRegressionNotEnoughLevelsException.assert_levels(cast(Sequence[Any], levels), input.target)

    # region Fitting
    regression = OrderedModel(Y.cat.codes, X, distr='logit')
    model = regression.fit(method='bfgs')
    self.logger.info(model.summary())

    # get dependent variables
    dependent_variable_levels: list[RegressionDependentVariableLevelInfo] = self._dependent_variable_levels(
      Y,
      reference_dependent=None,
      column=load_result.column
    )
    dependent_variable_level_names = list(map(lambda level: level.name, dependent_variable_levels))
    model_id = RegressionModelCacheManager().ordinal.save(RegressionModelCacheWrapper(
      model=model,
      levels=dependent_variable_level_names
    ))

    confidence_intervals = model.conf_int()
    results: list[OrdinalRegressionCoefficient] = []

    for col_idx, col in enumerate(X.columns):
      results.append(OrdinalRegressionCoefficient(
        name=str(col),

        value=model.params[col],
        std_err=model.bse[col],

        statistic=model.tvalues[col],
        p_value=model.pvalues[col],
        confidence_interval=confidence_intervals.loc[col, :],
      ))

    # The thresholds are not absolute thresholds, but rather increments.
    # https://www.statsmodels.org/dev/examples/notebooks/generated/ordinal_regression.html

    # len(levels) - 2 is safe. We already asserted the bounds above. Thresholds don't include the first and last element, so it's always -2.
    raw_threshold_increments = model.params[-(len(levels) - 2):]
    # Ignore the first and last element (that only contains -inf and inf)
    raw_thresholds = regression.transform_threshold_params(raw_threshold_increments)[1:-1]
    
    thresholds: list[OrdinalRegressionThreshold] = []
    for level_idx, raw_threshold in enumerate(raw_thresholds):
      thresholds.append(OrdinalRegressionThreshold(
        # This is also safe. Hopefully.
        from_level=str(levels[level_idx]),
        to_level=str(levels[level_idx + 1]),
        value=raw_threshold,
      ))

    # Get the remaining coefficient
    remaining_coefficient = self._calculate_remaining_coefficient(
      model=model,
      coefficients=model.params,
      preprocess=preprocess_result,
    )
    if remaining_coefficient is not None and preprocess_result.reference_idx is not None:
      results.insert(preprocess_result.reference_idx, OrdinalRegressionCoefficient.model_validate(
        remaining_coefficient,
        from_attributes=True,
      ))
    # Don't rename intercept to Baseline, ordinal regression doesn't have intercepts.

    # region Predictions
    # Order is same as coefficients (special case for GrandMeanDeviation!)
    model_prediction_input = self._regression_prediction_input(preprocess_result)
    model_predictions_probabilities = model.predict(
      model_prediction_input,
      which="prob"
    )
    model_predictions_latent_scores = model.predict(
      model_prediction_input,
      which="linpred"
    )
    model_predictions_cumulative_probabilities = model.predict(
      model_prediction_input,
      which="cumprob"
    )
    prediction_results = list(map(
      lambda coefficient, latent_score, probabilities, cumulative_probabilities: RegressionPredictionPerIndependentVariableResult(
        variable=coefficient.name if coefficient is not None else "Baseline",
        prediction=OrdinalRegressionPredictionResult(
          probabilities=probabilities,
          cumulative_probabilities=cumulative_probabilities,
          levels=dependent_variable_level_names,
          latent_score=latent_score
        )
      ),
      # Add "None" to represent the missing intercept. (the intercept originally represented the baseline)
      [None, *results],
      model_predictions_latent_scores,
      model_predictions_probabilities,
      model_predictions_cumulative_probabilities,
    ))
  
    return OrdinalRegressionResult(
      # Ordinal doesn't have intercept
      model_id=model_id,
      reference=preprocess_result.reference_name,
      independent_variables=preprocess_result.independent_variables,
      interpretation=input.interpretation,

      coefficients=results,
      thresholds=thresholds,
      levels=dependent_variable_levels,

      sample_size=len(Y),
      warnings=warnings,

      fit_evaluation=OrdinalRegressionFitEvaluation(
        log_likelihood_ratio=model.llr,
        p_value=model.llr_pvalue,
        pseudo_r_squared=model.prsquared,
        converged=model.mle_retvals.get('converged', True),
      ),
      predictions=prediction_results[1:],
      baseline_prediction=prediction_results[0].prediction,
    )
