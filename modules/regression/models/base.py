import abc
from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import SchemaColumn
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache
from modules.regression.exceptions import NoIndependentVariableDataException, RegressionInterpretationGrandMeanDeviationMutualExclusivityRequirementsViolationException, RegressionInterpretationRelativeToBaselineMutualExclusivityRequirementsViolationException, RegressionInterpretationRelativeToReferenceMutualExclusivityRequirementsViolationException, ReservedSubdatasetNameException
from modules.regression.models.utils import is_boolean_dataframe_mutually_exclusive, one_hot_to_effect_coding
from modules.regression.results.base import RegressionCoefficient, RegressionDependentVariableLevelInfo, RegressionIndependentVariableInfo, RegressionInterpretation
from modules.table.engine import TableEngine
from modules.table.filter_variants import NamedTableFilter, TableFilter

@dataclass
class RegressionProcessXResult:
  X: pd.DataFrame
  independent_variables: list[RegressionIndependentVariableInfo]
  reference: Optional[pd.Series]
  @property
  def reference_name(self)->Optional[str]:
    if self.reference is None:
      return None
    return str(self.reference.name)

@dataclass
class RegressionLoadResult:
  X: pd.DataFrame
  Y: pd.Series
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
    #   categorical_data = pd.Categorical(data, categories=categories, ordered=True)
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
      ))
    )

  def _process_X(self, X: pd.DataFrame, *, with_intercept: bool, interpretation: RegressionInterpretation, reference: Optional[str]):
    sample_sizes = X.astype(np.int32).sum(axis=0)
    independent_variable_names = list(map(str, X.columns))

    import statsmodels.api as sm
    reference_column: Optional[pd.Series] = None
    if interpretation == RegressionInterpretation.RelativeToReference:
      if reference is None:
        # Developer oversight
        raise ValueError(f"Provide a reference to _process_X when interpretation is {RegressionInterpretation.RelativeToReference.value}")
      # X must be mutually exclusive
      if not is_boolean_dataframe_mutually_exclusive(X):
        raise RegressionInterpretationRelativeToReferenceMutualExclusivityRequirementsViolationException()
      reference_column = X[reference]
      X.drop(columns=[reference], inplace=True)
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
      X = new_X
    else:
      raise ValueError(f"\"{interpretation}\" is not a valid regression interpretation type.")
    
    ReservedSubdatasetNameException.assert_column_names(list(map(str, X.columns)))
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = X.astype(np.float32)
    VIF = pd.Series([variance_inflation_factor(X, col_idx) for col_idx, col in enumerate(X.columns)], index=X.columns)
    if with_intercept:
      # Ignore coincidences where X has 1s already
      X = sm.add_constant(X, prepend=True, has_constant="add") # type: ignore
    X = X.astype(np.float32)

    independent_variables: list[RegressionIndependentVariableInfo] = []
    for name in independent_variable_names:
      independent_variables.append(RegressionIndependentVariableInfo(
        name=name,
        sample_size=sample_sizes[name] if name in sample_sizes.index else 0,
        variance_inflation_factor=VIF[name] if name in VIF.index else 1.0
      ))

    return RegressionProcessXResult(
      reference=reference_column,
      X=X,
      independent_variables=independent_variables,
    )
  
  def _calculate_remaining_coefficient(
    self,
    model: Any,
    coefficients: pd.Series,
    preprocess: RegressionProcessXResult,
    interpretation: RegressionInterpretation,
  )->Optional[RegressionCoefficient]:
    if interpretation != RegressionInterpretation.GrandMeanDeviation or preprocess.reference is None:
      return None
    reference_coefficient = -coefficients.iloc[1:].sum()
    # 0 to not test the intercept; -1 to test the remaining columns
    coefficient_weights = np.zeros_like(coefficients)
    coefficient_weights[0] = 0
    # It may be called a T-test, but the computation is based on the underlying model's distribution
    test_result = model.t_test(coefficient_weights)

    print(reference_coefficient, test_result.effect)

    return RegressionCoefficient(
      name=str(preprocess.reference.name),
      value=test_result.effect, # TODO: Sanity check coefficient weights with test_result.effect
      p_value=test_result.pvalue,
      std_err=test_result.sd,
      confidence_interval=test_result.conf_int()[0],
      statistic=test_result.statistic,
      # VIF doesn't make sense here.
    )

  def _regression_prediction_input(self, X: pd.DataFrame, *, interpretation: RegressionInterpretation):
    variable_count = len(X.columns)
    with_intercept = "const" in X.columns
    if with_intercept:
      variable_count -= 1

    prediction_input = np.eye(variable_count)
    if interpretation == RegressionInterpretation.GrandMeanDeviation:
      baseline_input = np.full((1, variable_count), -1)
    else:
      baseline_input = np.zeros((1, variable_count))

    prediction_input = np.vstack([baseline_input, prediction_input])

    if with_intercept:
      constants = np.ones((prediction_input.shape[0], 1))
      prediction_input = np.hstack([constants, prediction_input])
      
    return prediction_input
  
  def _dependent_variable_levels(self, Y: pd.Series, reference_dependent: Optional[str]):
    dependent_variable_levels: list[RegressionDependentVariableLevelInfo] = []
    for level in Y.cat.categories:
      sample_size = (Y == level).astype(np.int32).sum()
      level_name = str(level)
      if len(level_name) == 0 and reference_dependent is not None:
        level_name = reference_dependent

      dependent_variable_levels.append(RegressionDependentVariableLevelInfo(
        name=level_name,
        sample_size=sample_size,
      ))
    return dependent_variable_levels

  
  @property
  def logger(self):
    return ProvisionedLogger().provision("Regression")