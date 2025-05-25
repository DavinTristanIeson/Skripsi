import abc
from dataclasses import dataclass
from typing import Any, Optional, cast

import pandas as pd

from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import SchemaColumn
from modules.project.cache import ProjectCache
from modules.regression.exceptions import NoIndependentVariableDataException, RegressionInterpretationGrandMeanDeviationMutualExclusivityRequirementsViolationException, RegressionInterpretationRelativeToBaselineMutualExclusivityRequirementsViolationException, RegressionInterpretationRelativeToReferenceMutualExclusivityRequirementsViolationException
from modules.regression.models.utils import is_boolean_dataframe_mutually_exclusive, one_hot_to_effect_coding
from modules.regression.results.base import RegressionCoefficient, RegressionInterpretation
from modules.table.engine import TableEngine
from modules.table.filter_variants import NamedTableFilter, TableFilter

@dataclass
class RegressionProcessXResult:
  X: pd.DataFrame
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
      list(map(
        lambda group: engine
          .filter_mask(df, group.filter)
          .astype(pd.Float32Dtype()),
        groups
      )),
      columns=list(map(lambda group: group.name, groups))
    )

    if isinstance(target, str):
      Y = df[target]
      column = config.data_schema.assert_of_type(target, supported_types)
      mask = Y.notna()
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
      Y=Y
    )
  
  def _get_sample_sizes(self, X: pd.DataFrame, Y: pd.DataFrame):
    return X.sum(axis=0), len(Y)
  
  def _process_X(self, X: pd.DataFrame, *, with_intercept: bool, interpretation: RegressionInterpretation, reference: Optional[str]):
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
      new_X = one_hot_to_effect_coding(X, reference=reference)
      reference_column = X[reference] # mind the order
      X = new_X
    else:
      raise ValueError(f"\"{interpretation}\" is not a valid regression interpretation type.")

    X = X.astype(pd.Float32Dtype)
    if "const" in X.columns:
      raise ValueError(f"const is a reserved name. Please rename your subdataset.")
    if with_intercept:
      # Ignore coincidences where X has 1s already
      sm.add_constant(X, prepend=True, has_constant="add")
    return RegressionProcessXResult(
      reference=reference_column,
      X=X
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
    reference_coefficient = -coefficients.sum()
    # 0 to not test the intercept; -1 to test the remaining columns
    coefficient_weights = [0] + ([-1] * (len(coefficients) - 1))
    # It may be called a T-test, but the computation is based on the underlying model's distribution
    test_result = model.t_test(coefficient_weights)

    return RegressionCoefficient(
      name=str(preprocess.reference.name),
      value=test_result.effect[0], # TODO: Sanity check coefficient weights with test_result.effect
      sample_size=preprocess.reference.sum(),
      p_value=test_result.pvalue[0],
      std_err=test_result.sd[0],
      confidence_interval=test_result.conf_int(),
      statistic=test_result.statistic[0],
      # VIF doesn't make sense here.
      variance_inflation_factor=0.0
    )