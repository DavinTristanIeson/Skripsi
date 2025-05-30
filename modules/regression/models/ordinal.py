from dataclasses import dataclass
from typing import Any, Sequence, cast
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

from modules.config.schema.base import ORDERED_CATEGORICAL_SCHEMA_COLUMN_TYPES
from modules.regression.exceptions import OrdinalRegressionNotEnoughLevelsException
from modules.regression.models.base import BaseRegressionModel
from modules.regression.models.cache import RegressionModelCacheManager
from modules.regression.results.ordinal import OrdinalRegressionCoefficient, OrdinalRegressionInput, OrdinalRegressionLevelSampleSize, OrdinalRegressionThreshold, OrdinalRegressionResult

@dataclass
class OrdinalRegressionModel(BaseRegressionModel):
  input: OrdinalRegressionInput

  def fit(self):
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

    Y = Y.cat.remove_unused_categories()
    levels = Y.cat.categories
    OrdinalRegressionNotEnoughLevelsException.assert_levels(cast(Sequence[Any], levels), input.target)

    regression = OrderedModel(Y.cat.codes, X, distr='logit')
    model = regression.fit(method='bfgs')
    self.logger.info(model.summary())
    model_id = RegressionModelCacheManager().ordinal.save(model) # type: ignore

    confidence_intervals = model.conf_int()
    results: list[OrdinalRegressionCoefficient] = []

    for col_idx, col in enumerate(X.columns):
      results.append(OrdinalRegressionCoefficient(
        name=str(col),

        value=model.params[col],
        std_err=model.bse[col],
        sample_size=preprocess_result.sample_sizes[col],

        statistic=model.tvalues[col],
        p_value=model.pvalues[col],
        confidence_interval=confidence_intervals.loc[col, :],
        
        variance_inflation_factor=variance_inflation_factor(X.values, col_idx),
      ))

    # The thresholds are not absolute thresholds, but rather increments.
    # https://www.statsmodels.org/dev/examples/notebooks/generated/ordinal_regression.html

    # len(levels) - 2 is safe. We already asserted the bounds above. Thresholds don't include the first and last element, so it's always -2.
    raw_threshold_increments = model.params[-(len(levels) - 2):]
    # Ignore the first and last element (that only contains -inf and inf)
    raw_thresholds = regression.transform_threshold_params(raw_threshold_increments)[1:-1]

    sample_sizes: list[OrdinalRegressionLevelSampleSize] = []
    for level in levels:
      sample_size = (Y == level).sum()
      sample_sizes.append(OrdinalRegressionLevelSampleSize(
        name=str(level),
        sample_size=sample_size,
      ))
    
    thresholds: list[OrdinalRegressionThreshold] = []
    for level_idx, raw_threshold in enumerate(raw_thresholds):
      idx = len(X.columns) + level_idx
      thresholds.append(OrdinalRegressionThreshold(
        # This is also safe. Hopefully.
        from_level=levels[level_idx],
        to_level=levels[level_idx + 1],
        value=raw_threshold,
      ))

    remaining_coefficient = self._calculate_remaining_coefficient(
      model=model,
      coefficients=model.params,
      interpretation=input.interpretation,
      preprocess=preprocess_result,
    )
    if remaining_coefficient is not None:
      results.append(OrdinalRegressionCoefficient.model_validate(
        remaining_coefficient,
        from_attributes=True,
      ))

    return OrdinalRegressionResult(
      model_id=model_id,
      reference=preprocess_result.reference_name,
      interpretation=input.interpretation,
      sample_sizes=sample_sizes,
      coefficients=results,
      thresholds=thresholds,
      log_likelihood_ratio=model.llr,
      p_value=model.llr_pvalue,
      pseudo_r_squared=model.prsquared,
      converged=model.mle_retvals.get('converged', True),
      warnings=warnings,
      sample_size=len(Y),
    )
