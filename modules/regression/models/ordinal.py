from dataclasses import dataclass
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

from modules.config.schema.base import ORDERED_CATEGORICAL_SCHEMA_COLUMN_TYPES
from modules.logger.provisioner import ProvisionedLogger
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

    model = OrderedModel(Y.cat.codes, X, distr='logit').fit(method='bfgs')
    self.logger.info(model.summary())

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
    cutpoints: list[OrdinalRegressionCutpoint] = []
    for idx in range(len(X.columns), len(model.params)):
      category_idx = idx - len(X.columns)
      category = Y.cat.categories[category_idx]
      cutpoints.append(OrdinalRegressionCutpoint(
        name=str(category),
        value=model.params.iloc[idx],
        std_err=model.bse.iloc[idx],
        sample_size=(Y == category).sum(),
        confidence_interval=confidence_intervals.iloc[idx],
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
      p_value=model.llr_pvalue,
      pseudo_r_squared=model.prsquared,
      converged=model.mle_retvals.get('converged', True),
      warnings=warnings,
      sample_size=len(Y),
    )
