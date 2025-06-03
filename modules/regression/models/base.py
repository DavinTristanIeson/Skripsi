import abc
from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import SchemaColumn, TemporalPrecisionEnum
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache
from modules.regression.exceptions import NoIndependentVariableDataException, RegressionInterpretationGrandMeanDeviationMutualExclusivityRequirementsViolationException, RegressionInterpretationRelativeToBaselineMutualExclusivityRequirementsViolationException, RegressionInterpretationRelativeToReferenceMutualExclusivityRequirementsViolationException, ReservedSubdatasetNameException
from modules.regression.models.utils import is_boolean_dataframe_mutually_exclusive, one_hot_to_effect_coding
from modules.regression.results.base import RegressionCoefficient, RegressionDependentVariableLevelInfo, RegressionIndependentVariableInfo, RegressionInterpretation
from modules.table.engine import TableEngine
from modules.table.filter_variants import NamedTableFilter

@dataclass
class RegressionProcessXResult:
  X: pd.DataFrame
  independent_variables: list[RegressionIndependentVariableInfo]
  reference: Optional[pd.Series]
  reference_idx: Optional[int]
  interpretation: RegressionInterpretation
  @property
  def reference_name(self)->Optional[str]:
    if self.reference is None:
      return None
    return str(self.reference.name)

@dataclass
class RegressionLoadResult:
  X: pd.DataFrame
  Y: pd.Series
  column: Optional[SchemaColumn]
  independent_variables: list[str]

@dataclass
class BaseRegressionModel(abc.ABC):
  cache: ProjectCache

  def _transform_data(self, data: pd.Series, column: Optional[SchemaColumn]):
    if column is None:
      return data
    if column.type == SchemaColumnTypeEnum.Topic:
      tm_result = self.cache.topics.load(cast(str, column.source_name))
      categorical_data = pd.Categorical(data)
      categorical_data = categorical_data.rename_categories(tm_result.renamer)
      return pd.Series(categorical_data, name=column.name)
    if column.type == SchemaColumnTypeEnum.Boolean:
      categorical_data = pd.Categorical(data)
      categorical_data = categorical_data.rename_categories({
        True: "True",
        False: "False"
      })
      return pd.Series(categorical_data, name=column.name)
    if column.type == SchemaColumnTypeEnum.Temporal:
      cat_Y_categories = pd.Categorical(data.sort_values(), ordered=True).categories
      cat_Y = pd.Categorical(data, categories=cat_Y_categories, ordered=True)
      return pd.Series(cat_Y, index=data.index)
    # if column.type == SchemaColumnTypeEnum.Temporal and column.internal:
    #   if column.temporal_precision == TemporalPrecisionEnum.Year:
    #     strftime_format = "%Y"
    #   elif column.temporal_precision == TemporalPrecisionEnum.Month:
    #     strftime_format = "%m %Y"
    #   elif column.temporal_precision == TemporalPrecisionEnum.Date:
    #     strftime_format = "%d %m %Y"
    #   else:
    #     return data
    #   categories = data.sort_values().dt.strftime(strftime_format).unique()
    #   # Messes with reference_dependent.
    #   categorical_data = pd.Categorical(data.dt.strftime(strftime_format), categories=categories, ordered=True)
    #   return pd.Series(categorical_data, name=column.name)
    return data
  
  def _load(self, groups: list[NamedTableFilter], target: str | NamedTableFilter, constrain_by_X: bool, supported_types: list[SchemaColumnTypeEnum], transform_data: bool = False):
    cache = self.cache
    config = cache.config

    df = cache.workspaces.load()
    engine = TableEngine(config=cache.config)
    X = pd.DataFrame(
      dict(map(
        lambda group: (group.name, engine
          .filter_mask(df, group.filter)),
        groups
      )),
      dtype=pd.BooleanDtype(),
    )
    X = X.fillna(False)

    column: Optional[SchemaColumn] = None
    if isinstance(target, str):
      Y = df[target]
      column = config.data_schema.assert_of_type(target, supported_types)
      mask = Y.notna()
      X = X.loc[mask, :]
      Y = Y[mask]
      if column.type == SchemaColumnTypeEnum.Topic:
        mask = mask & (Y != -1)
    else:
      Y = TableEngine(config=cache.config).filter_mask(df, target.filter)
      Y.name = target.name
      column = None

    if len(Y) == 0:
      raise NoIndependentVariableDataException(
        column=column.name if column is not None else str(Y.name)
      )
    
    if constrain_by_X:
      X_agg = X.any(axis=1)
      X = X.loc[X_agg, :]
      Y = Y[X_agg]

    # Use categorical dtype for topic
    Y = self._transform_data(Y, column)
    return RegressionLoadResult(
      X=X,
      Y=Y,
      independent_variables=list(map(
        lambda group: group.name, groups
      )),
      column=column
    )

  def _process_X(self, X: pd.DataFrame, *, with_intercept: bool, interpretation: RegressionInterpretation, reference: Optional[str]):
    # X is guaranteed to be a boolean dataframe, so using .sum() is natural.
    # This might be a headache if we want to support numbers in the future.
    sample_sizes = X.astype(np.int32).sum(axis=0)
    # Get the names of the independent variables for easy enumeration
    independent_variable_names = list(map(str, X.columns))

    import statsmodels.api as sm

    # Reference column will be excluded from the design matrix, but we still want to keep it to calculate things like the remaining
    # coefficient for effect coding.
    reference_column: Optional[pd.Series] = None
    # Reference IDX is necessary to insert the reference coefficient (effect coding) into the appropriate place.
    # Technically we can just move it to the end, but I want to preserve the order (makes things so much more complicated though)
    reference_idx: Optional[int] = None
    if interpretation == RegressionInterpretation.RelativeToReference:
      if reference is None:
        # Developer oversight
        raise ValueError(f"Provide a reference to _process_X when interpretation is {RegressionInterpretation.RelativeToReference.value}")
      # X must be mutually exclusive
      if not is_boolean_dataframe_mutually_exclusive(X):
        raise RegressionInterpretationRelativeToReferenceMutualExclusivityRequirementsViolationException()
      reference_column = X[reference]
      reference_idx = list(X.columns).index(reference)
      # Remove ref
      X = X.drop(columns=[reference])
    elif interpretation == RegressionInterpretation.RelativeToBaseline:
      # X must not be mutually exclusive
      if is_boolean_dataframe_mutually_exclusive(X):
        raise RegressionInterpretationRelativeToBaselineMutualExclusivityRequirementsViolationException()
    elif interpretation == RegressionInterpretation.GrandMeanDeviation:
      # X must be mutually exclusive
      if not is_boolean_dataframe_mutually_exclusive(X):
        raise RegressionInterpretationGrandMeanDeviationMutualExclusivityRequirementsViolationException()
      reference = reference or X.columns[-1]
      new_X = one_hot_to_effect_coding(X, reference=reference)
      reference_column = X[reference] # mind the order
      reference_idx = list(X.columns).index(reference)
      X = new_X
    else:
      raise ValueError(f"\"{interpretation}\" is not a valid regression interpretation type.")
    
    # make sure there's no reserved names
    ReservedSubdatasetNameException.assert_column_names(list(map(str, X.columns)))
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # Add constant so that we can calculate VIF
    X = X.astype(np.float32)
    X = sm.add_constant(X, prepend=True, has_constant="add") # type: ignore
    # VIF expects a constant column, so this is necessary.
    VIF = pd.Series([variance_inflation_factor(X, col_idx) for col_idx, col in enumerate(X.columns)], index=X.columns)
    if not with_intercept:
      # remove const if we don't need intercept (for ordinal regression)
      X = X.drop(columns=["const"])
    
    # Get independent variable info (note that independent variable info order should not be conflated with the coefficients due to the reference column shuffling up there)
    independent_variables: list[RegressionIndependentVariableInfo] = []
    for name in independent_variable_names:
      independent_variables.append(RegressionIndependentVariableInfo(
        name=name,
        sample_size=sample_sizes[name] if name in sample_sizes.index else 0,
        variance_inflation_factor=VIF[name] if name in VIF.index else 1.0
      ))

    return RegressionProcessXResult(
      reference=reference_column,
      reference_idx=reference_idx,
      X=X,
      interpretation=interpretation,
      independent_variables=independent_variables,
    )
  
  def _calculate_remaining_coefficient(
    self,
    model: Any,
    coefficients: pd.Series,
    preprocess: RegressionProcessXResult,
  )->Optional[RegressionCoefficient]:
    if preprocess.interpretation != RegressionInterpretation.GrandMeanDeviation or preprocess.reference is None:
      return None
    reference_coefficient = -coefficients.iloc[1:].sum()
    # 0 to not test the intercept; -1 to test the remaining columns
    coefficient_weights = np.zeros_like(coefficients)
    coefficient_weights[1:] = -1
    # It may be called a T-test, but the computation is based on the underlying model's distribution
    test_result = model.t_test(coefficient_weights)

    return RegressionCoefficient(
      name=str(preprocess.reference.name),
      value=test_result.effect, # Sanity check coefficient weights with test_result.effect
      p_value=test_result.pvalue,
      std_err=test_result.sd,
      confidence_interval=test_result.conf_int()[0],
      statistic=test_result.statistic,
    )

  def _regression_prediction_input(self, preprocess: RegressionProcessXResult):
    X = preprocess.X
    variable_count = len(X.columns)
    with_intercept = "const" in X.columns
    if with_intercept:
      variable_count -= 1

    prediction_input = np.eye(variable_count)
    baseline_input = np.zeros((1, variable_count))
    if preprocess.interpretation == RegressionInterpretation.GrandMeanDeviation and preprocess.reference_idx is not None:
      prediction_input = np.vstack([
        prediction_input[:preprocess.reference_idx],
        # Insert a row for the reference variable
        np.full((1, variable_count), -1),
        prediction_input[preprocess.reference_idx+1:],
      ])
    # Other interpretations don't need to have the reference re-added.
    
      
    # Add baseline and intercept
    prediction_input = np.vstack([baseline_input, prediction_input])
    if with_intercept:
      constants = np.ones((prediction_input.shape[0], 1))
      prediction_input = np.hstack([constants, prediction_input])

    return prediction_input
  
  def _dependent_variable_levels(self, Y: pd.Series, reference_dependent: Optional[str], column: Optional[SchemaColumn]):
    # get the dependent variable level info
    dependent_variable_levels: list[RegressionDependentVariableLevelInfo] = []
    for level in Y.cat.categories:
      sample_size = (Y == level).astype(np.int32).sum()
      level_name = str(level)
      if len(level_name) == 0 and reference_dependent is not None:
        level_name = reference_dependent

      if column is not None and column.type == SchemaColumnTypeEnum.Temporal and column.internal and pd.api.types.is_datetime64_any_dtype(level):
        # only format when temporal precision exists and level is datetime64.
        strftime_format: Optional[str] = None
        if column.temporal_precision == TemporalPrecisionEnum.Year:
          strftime_format = "%Y"
        elif column.temporal_precision == TemporalPrecisionEnum.Month:
          strftime_format = "%m %Y"
        elif column.temporal_precision == TemporalPrecisionEnum.Date:
          strftime_format = "%d %m %Y"
        else:
          pass

        if strftime_format is not None:
          try:
            dt_python = level.astype('datetime64[s]').tolist()
            level_name = dt_python.strftime(strftime_format)
          except Exception:
            pass
    #   categories = data.sort_values().dt.strftime(strftime_format).unique()
    #   # Messes with reference_dependent.
    #   categorical_data = pd.Categorical(data.dt.strftime(strftime_format), categories=categories, ordered=True)
    #   return pd.Series(categorical_data, name=column.name)

      dependent_variable_levels.append(RegressionDependentVariableLevelInfo(
        name=level_name,
        sample_size=sample_size,
      ))
    return dependent_variable_levels

  
  @property
  def logger(self):
    return ProvisionedLogger().provision("Regression")