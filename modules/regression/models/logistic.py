from dataclasses import dataclass
from typing import Any, Optional, cast
import numpy as np
import pandas as pd
from modules.config.schema.base import CATEGORICAL_SCHEMA_COLUMN_TYPES, SchemaColumnTypeEnum
from modules.logger.provisioner import ProvisionedLogger
from modules.regression.exceptions import DependentVariableReferenceMustBeAValidValueException
from modules.regression.models.base import BaseRegressionModel, RegressionProcessXResult
from modules.regression.models.cache import RegressionModelCacheManager, RegressionModelCacheWrapper
from modules.regression.results.base import RegressionCoefficient, RegressionInterpretation, RegressionDependentVariableLevelInfo, RegressionPredictionPerIndependentVariableResult
from modules.regression.results.logistic import LogisticRegressionCoefficient, LogisticRegressionFitEvaluation, LogisticRegressionInput, LogisticRegressionPredictionResult, MultinomialLogisticRegressionFacetResult, MultinomialLogisticRegressionPredictionResult, MultinomialLogisticRegressionResult, LogisticRegressionResult, MultinomialLogisticRegressionInput

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

    # region Fit Model
    model = sm.Logit(Y, X).fit(maxiter=100, method="bfgs")
    self.logger.info(model.summary())
    model_id = RegressionModelCacheManager().logistic.save(RegressionModelCacheWrapper(
      model=model,
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

    return LogisticRegressionResult(
      model_id=model_id,
      reference=preprocess_result.reference_name,
      independent_variables=preprocess_result.independent_variables,
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
      predictions=prediction_results[1:],
      baseline_prediction=prediction_results[0].prediction,
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

    import statsmodels.api as sm
    # Newton solver produces NaN too often.
    model = sm.MNLogit(Y, X).fit(maxiter=500, method="bfgs")
    self.logger.info(model.summary())

    # Get dependent variable levels
    dependent_variable_levels: list[RegressionDependentVariableLevelInfo] = self._dependent_variable_levels(
      Y, reference_dependent=reference_dependent
    )
    model_id = RegressionModelCacheManager().multinomial_logistic.save(RegressionModelCacheWrapper(
      model=model,
      # Store the levels since statsmodels doesn't.
      levels=list(map(lambda level: level.name, dependent_variable_levels))
    ))
    
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
          confidence_interval=confidence_intervals.loc[(row, col)].to_list(),
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

  
    return MultinomialLogisticRegressionResult(
      model_id=model_id,
      reference=preprocess_result.reference_name,
      reference_dependent=reference_dependent,
      levels=dependent_variable_levels,
      independent_variables=preprocess_result.independent_variables,
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
      predictions=prediction_results[1:],
      baseline_prediction=prediction_results[0].prediction,
    )
