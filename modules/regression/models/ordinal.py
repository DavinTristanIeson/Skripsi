import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

from modules.config.schema.base import ORDERED_CATEGORICAL_SCHEMA_COLUMN_TYPES
from modules.regression.models.base import BaseRegressionModel
from modules.regression.results.base import BaseRegressionInput, RegressionCoefficient
from modules.regression.results.ordinal import OrdinalRegressionCutpoint, OrdinalRegressionResult


class OrdinalRegressionModel(BaseRegressionModel):
  input: BaseRegressionInput

  def fit(self):
    input = self.input

    X, Y = self._load(
      groups=input.groups,
      target=input.target,
      constrain_by_X=input.constrain_by_groups,
      supported_types=ORDERED_CATEGORICAL_SCHEMA_COLUMN_TYPES
    )
    X = self._process_X(X, with_intercept=False, interpretation=input.interpretation, reference=input.reference)
    warnings = []

    model = OrderedModel(Y, X, distr='logit')
    result = model.fit(method='bfgs', disp=0)

    confidence_intervals = result.conf_int()
    results: list[RegressionCoefficient] = []

    for col_idx, col in enumerate(X.columns):
      results.append(RegressionCoefficient(
        name=str(col),
        value=result.params[col],
        std_err=result.bse[col],
        sample_size=X[col].sum(),
        statistic=result.zvalues[col],
        p_value=result.pvalues[col],
        confidence_interval=confidence_intervals.loc[col].tolist(),
        variance_inflation_factor=variance_inflation_factor(X.values, col_idx),
      ))
    cutpoint_names = set(result.params.index) - set(X.columns)
    cutpoints: list[OrdinalRegressionCutpoint] = []
    for col in cutpoint_names:
      cutpoints.append(OrdinalRegressionCutpoint(
        name=str(col),
        value=result.params[col],
        std_err=result.bse[col],
      ))

    return OrdinalRegressionResult(
      reference=input.reference,
      interpretation=input.interpretation,
      coefficients=results,
      cutpoints=cutpoints,
      log_likelihood_ratio=result.llr,
      converged=result.mle_retvals.get('converged', True),
      warnings=warnings,
    )
