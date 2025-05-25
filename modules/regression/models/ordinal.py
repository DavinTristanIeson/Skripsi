from dataclasses import dataclass
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

from modules.config.schema.base import ORDERED_CATEGORICAL_SCHEMA_COLUMN_TYPES
from modules.regression.models.base import BaseRegressionModel
from modules.regression.results.base import BaseRegressionInput
from modules.regression.results.ordinal import OrdinalRegressionCoefficient, OrdinalRegressionCutpoint, OrdinalRegressionResult

@dataclass
class OrdinalRegressionModel(BaseRegressionModel):
  input: BaseRegressionInput

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

    model = OrderedModel(Y, X, distr='logit').fit(method='bfgs')

    confidence_intervals = model.conf_int()
    results: list[OrdinalRegressionCoefficient] = []

    for col_idx, col in enumerate(X.columns):
      results.append(OrdinalRegressionCoefficient(
        name=str(col),

        value=model.params[col],
        std_err=model.bse[col],
        sample_size=X[col].sum(),

        statistic=model.zvalues[col],
        p_value=model.pvalues[col],
        confidence_interval=confidence_intervals.loc[col].tolist(),
        
        variance_inflation_factor=variance_inflation_factor(X.values, col_idx),
      ))
    cutpoint_names = set(model.params.index) - set(X.columns)
    cutpoints: list[OrdinalRegressionCutpoint] = []
    for col in cutpoint_names:
      cutpoints.append(OrdinalRegressionCutpoint(
        name=str(col),
        value=model.params[col],
        std_err=model.bse[col],
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
      reference=preprocess_result.reference_name,
      interpretation=input.interpretation,
      coefficients=results,
      cutpoints=cutpoints,
      log_likelihood_ratio=model.llr,
      converged=model.mle_retvals.get('converged', True),
      warnings=warnings,
      sample_size=len(Y),
    )
