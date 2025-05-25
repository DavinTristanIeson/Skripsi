from dataclasses import dataclass
import pandas as pd
from modules.config.schema.base import CATEGORICAL_SCHEMA_COLUMN_TYPES, SchemaColumnTypeEnum
from modules.regression.exceptions import DependentVariableReferenceMustBeAValidValueException
from modules.regression.models.base import BaseRegressionModel
from modules.regression.results.logistic import LogisticRegressionCoefficient, LogisticRegressionInput, MultinomialLogisticRegressionFacetResult, MultinomialLogisticRegressionResult, LogisticRegressionResult, MultinomialLogisticRegressionInput

@dataclass
class LogisticRegressionModel(BaseRegressionModel):
  input: LogisticRegressionInput

  def fit(self):
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
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    model = sm.Logit(Y, X).fit()
    confidence_intervals = model.conf_int()

    results: list[LogisticRegressionCoefficient] = []
    for col_idx, col in enumerate(X.columns):
      results.append(LogisticRegressionCoefficient(
        name=str(col),
        value=model.params[col],
        std_err=model.bse[col],
        sample_size=X[col].sum(),

        statistic=model.zvalues[col],
        p_value=model.pvalues[col],
        confidence_interval=confidence_intervals.loc[col].to_list(),
        variance_inflation_factor=variance_inflation_factor(X.values, col_idx),
      ))
    
    remaining_coefficient = self._calculate_remaining_coefficient(
      model=model,
      coefficients=model.params,
      interpretation=input.interpretation,
      preprocess=preprocess_result,
    )
    if remaining_coefficient is not None:
      results.append(LogisticRegressionCoefficient.model_validate(
        remaining_coefficient,
        from_attributes=True,
      ))

    return LogisticRegressionResult(
      converged=model.mle_retvals.get('converged', True),
      reference=preprocess_result.reference_name,
      intercept = results[0],
      coefficients = results[1:],
      p_value=model.llr_pvalue,
      pseudo_r_squared=model.prsquared,
      log_likelihood_ratio=model.llr,
      interpretation=input.interpretation,
      sample_size=len(Y),
      warnings=[],
    )


@dataclass
class MultinomialLogisticRegressionModel(BaseRegressionModel):
  input: MultinomialLogisticRegressionInput

  def fit(self):
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

    warnings = []

    Y = pd.Categorical(Y)
    if input.reference_dependent is not None:
      if input.reference_dependent not in Y.categories:
        raise DependentVariableReferenceMustBeAValidValueException(reference=input.reference_dependent, supported_values=Y.categories)
      Y_categories = list(Y.categories)
      idx = Y_categories.index(input.reference_dependent)
      if idx == -1:
        raise ValueError("Can't find the category that corresponds to reference dependent. This might be a developer oversight.")
      category = Y_categories.pop(idx)
      Y_categories.insert(0, category)
      # make the reference first so that the model automatically uses it as the baseline.
      Y.reorder_categories(Y_categories)

    
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    model = sm.MNLogit(Y, X).fit()
    confidence_intervals = model.conf_int()

    facets: list[MultinomialLogisticRegressionFacetResult] = []
    for row_idx, row in enumerate(Y.categories):
      results: list[LogisticRegressionCoefficient] = []
      for col_idx, col in enumerate(X.columns):
        results.append(LogisticRegressionCoefficient(
          name=str(col),

          value=model.params[col],
          std_err=model.bse[col],
          sample_size=X[col].sum(),

          statistic=model.zvalues[col],
          p_value=model.pvalues[col],
          confidence_interval=confidence_intervals.loc[col].to_list(),
          
          variance_inflation_factor=variance_inflation_factor(X.values, col_idx),
        ))
      # This will definitely have issues due to how MNLogit works. But that's for future me to debug.
      remaining_coefficient = self._calculate_remaining_coefficient(
        model=model,
        coefficients=model.params,
        interpretation=input.interpretation,
        preprocess=preprocess_result,
      )
      if remaining_coefficient is not None:
        results.append(LogisticRegressionCoefficient.model_validate(
          remaining_coefficient,
          from_attributes=True,
        ))
      facets.append(MultinomialLogisticRegressionFacetResult(
        coefficients=results[1:],
        intercept=results[0],
      ))

    return MultinomialLogisticRegressionResult(
      facets=facets,
      converged=model.mle_retvals.get('converged', True),
      reference=input.reference,
      p_value=model.llr_pvalue,
      pseudo_r_squared=model.prsquared,
      log_likelihood_ratio=model.llr,
      interpretation=input.interpretation,
      sample_size=len(Y),
      warnings=warnings,
    )
