from dataclasses import dataclass
from typing import Any, Optional, cast
import numpy as np
import pandas as pd
from modules.config.schema.base import CATEGORICAL_SCHEMA_COLUMN_TYPES, SchemaColumnTypeEnum
from modules.logger.provisioner import ProvisionedLogger
from modules.regression.exceptions import DependentVariableReferenceMustBeAValidValueException
from modules.regression.models.base import BaseRegressionModel, RegressionProcessXResult
from modules.regression.models.cache import RegressionModelCacheManager
from modules.regression.results.base import RegressionCoefficient, RegressionInterpretation, RegressionPredictionPerIndependentVariableResult
from modules.regression.results.logistic import LogisticRegressionCoefficient, LogisticRegressionFitEvaluation, LogisticRegressionInput, LogisticRegressionPredictionResult, MultinomialLogisticRegressionFacetResult, MultinomialLogisticRegressionPredictionResult, MultinomialLogisticRegressionResult, LogisticRegressionResult, MultinomialLogisticRegressionInput

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

    model = sm.Logit(Y, X).fit(maxiter=100, method="bfgs")
    self.logger.info(model.summary())
    model_id = RegressionModelCacheManager().logistic.save(model) # type: ignore

    confidence_intervals = model.conf_int()

    results: list[LogisticRegressionCoefficient] = []
    for col_idx, col in enumerate(X.columns):
      results.append(LogisticRegressionCoefficient(
        name=str(col),
        value=model.params[col],
        std_err=model.bse[col],
        sample_size=preprocess_result.sample_sizes[col],

        statistic=model.tvalues[col],
        p_value=model.pvalues[col],
        confidence_interval=confidence_intervals.loc[col, :],
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

    model_predictions = model.predict(
      self._regression_prediction_input(load_result.independent_variables)
    )

    prediction_results = list(map(
      lambda variable, result: RegressionPredictionPerIndependentVariableResult(
        variable=variable,
        prediction=LogisticRegressionPredictionResult(
          probability=result
        )
      ),
      load_result.independent_variables,
      model_predictions
    ))

    return LogisticRegressionResult(
      model_id=model_id,
      reference=preprocess_result.reference_name,
      independent_variables=load_result.independent_variables,
      interpretation=input.interpretation,

      intercept = results[0],
      coefficients = results[1:],

      sample_size=len(Y),
      warnings=[],

      fit_evaluation=LogisticRegressionFitEvaluation(
        converged=model.mle_retvals.get('converged', True),
        p_value=model.llr_pvalue,
        pseudo_r_squared=model.prsquared,
        log_likelihood_ratio=model.llr,
      ),
      predictions=prediction_results,
    )

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
      sample_size=int(preprocess.reference.sum()),
      p_value=test_result.pvalue,
      std_err=test_result.sd[0, 0],
      confidence_interval=test_result.conf_int()[0].tolist(),
      statistic=test_result.statistic[0, 0],
      # VIF doesn't make sense here.
      variance_inflation_factor=0.0
    )
  

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
    original_X = load_result.X
    X = preprocess_result.X
    Y = load_result.Y

    warnings = []

    Y = pd.Categorical(Y)
    
    reference_dependent = input.reference_dependent or Y.categories[0]
    if reference_dependent not in Y.categories:
      raise DependentVariableReferenceMustBeAValidValueException(reference=reference_dependent, supported_values=Y.categories)
    Y_categories = list(Y.categories)
    idx = Y_categories.index(input.reference_dependent)
    if idx == -1:
      raise ValueError("Can't find the category that corresponds to reference dependent. This might be a developer oversight.")
    category = Y_categories.pop(idx)
    Y_categories.insert(0, category)
    # make the reference first so that the model automatically uses it as the baseline.
    Y = Y.reorder_categories(Y_categories)
    # Make sure that the category is first
    Y = Y.rename_categories({
      # Ensure that this is first alphabetically
      input.reference_dependent: "",
    })
    
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Newton solver produces NaN too often.
    model = sm.MNLogit(Y, X).fit(maxiter=500, method="bfgs")
    self.logger.info(model.summary())
    model_id = RegressionModelCacheManager().multinomial_logistic.save(model) # type: ignore
    
    confidence_intervals = model.conf_int()

    facets: list[MultinomialLogisticRegressionFacetResult] = []
    # First category is excluded
    for row_idx, row in enumerate(Y.categories[1:]):
      results: list[LogisticRegressionCoefficient] = []
      for col_idx, col in enumerate(X.columns):
        if col == "const":
          sample_size = (Y == row).sum()
        else:
          sample_size = cast(pd.Series, original_X.loc[Y == row, col]).sum()
          
        VIF = variance_inflation_factor(X.values, col_idx)
        results.append(LogisticRegressionCoefficient(
          name=str(col),

          value=model.params.at[col, row_idx],
          std_err=model.bse.at[col, row_idx],
          sample_size=int(sample_size),

          statistic=model.tvalues.at[col, row_idx],
          p_value=model.pvalues.at[col, row_idx],
          confidence_interval=confidence_intervals.loc[(row, col)].to_list(),
          
          variance_inflation_factor=VIF,
        ))
      # This will definitely have issues due to how MNLogit works. But that's for future me to debug.
      remaining_coefficient = self._calculate_remaining_coefficient_logistic(
        model=model,
        coefficients=model.params,
        interpretation=input.interpretation,
        preprocess=preprocess_result,
        idx=row_idx,
      )
      if remaining_coefficient is not None:
        results.append(LogisticRegressionCoefficient.model_validate(
          remaining_coefficient,
          from_attributes=True,
        ))
      intercept = results[0]
      intercept.name = row
      intercept.sample_size = int((Y == row).sum())
      facets.append(MultinomialLogisticRegressionFacetResult(
        level=row,
        coefficients=results[1:],
        intercept=intercept,
      ))

    model_predictions = model.predict(
      self._regression_prediction_input(load_result.independent_variables)
    )
    prediction_results = list(map(
      lambda variable, result: RegressionPredictionPerIndependentVariableResult(
        variable=variable,
        prediction=MultinomialLogisticRegressionPredictionResult(
          probabilities=result,
          levels=result.columns,
        )
      ),
      load_result.independent_variables,
      model_predictions.iterrows()
    ))
  
    return MultinomialLogisticRegressionResult(
      model_id=model_id,
      reference=preprocess_result.reference_name,
      reference_dependent=reference_dependent,
      levels=list(map(lambda facet: facet.level, facets)),
      independent_variables=load_result.independent_variables,
      interpretation=input.interpretation,

      facets=facets,

      sample_size=len(Y),
      warnings=[],

      fit_evaluation=LogisticRegressionFitEvaluation(
        converged=model.mle_retvals.get('converged', True),
        p_value=model.llr_pvalue,
        pseudo_r_squared=model.prsquared,
        log_likelihood_ratio=model.llr,
      ),
      predictions=prediction_results,
    )
