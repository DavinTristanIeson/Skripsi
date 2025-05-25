import abc
from dataclasses import dataclass
from typing import Optional, cast

import pandas as pd

from modules.config.schema.base import ANALYZABLE_SCHEMA_COLUMN_TYPES, SchemaColumnTypeEnum
from modules.config.schema.schema_variants import SchemaColumn
from modules.project.cache import ProjectCache
from modules.regression.exceptions import NoIndependentVariableDataException, RegressionInterpretationGrandMeanDeviationMutualExclusivityRequirementsViolationException, RegressionInterpretationRelativeToBaselineMutualExclusivityRequirementsViolationException, RegressionInterpretationRelativeToReferenceMutualExclusivityRequirementsViolationException
from modules.regression.models.utils import is_boolean_dataframe_mutually_exclusive, one_hot_to_effect_coding
from modules.regression.results.base import RegressionInterpretation
from modules.table.engine import TableEngine
from modules.table.filter_variants import NamedTableFilter

@dataclass
class BaseRegressionModel(abc.ABC):
  cache: ProjectCache

  def _transform_data(self, data: pd.Series, column: SchemaColumn):
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
    return data
  
  def _load(self, groups: list[NamedTableFilter], target: str, constrain_by_X: bool, supported_types: list[SchemaColumnTypeEnum], transform_data: bool = False):
    cache = self.cache
    config = cache.config
    column = config.data_schema.assert_of_type(target, supported_types)

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
    Y = df[target]

    mask = Y.notna()
    if column.type == SchemaColumnTypeEnum.Topic:
      mask = mask & (Y != -1)

    if len(Y) == 0:
      raise NoIndependentVariableDataException(column=column.name)
    
    if constrain_by_X:
      X_agg = X.any(axis=1)
      X = X.loc[X_agg, :]
      Y = Y[X_agg]

    # Use categorical dtype for topic
    Y = self._transform_data(Y, column)
    return X, Y
  
  def _get_sample_sizes(self, X: pd.DataFrame, Y: pd.DataFrame):
    return X.sum(axis=0), len(Y)
  
  def _process_X(self, X: pd.DataFrame, *, with_intercept: bool, interpretation: RegressionInterpretation, reference: Optional[str]):
    import statsmodels.api as sm
    if interpretation == RegressionInterpretation.RelativeToReference:
      if reference is None:
        # Developer oversight
        raise ValueError(f"Provide a reference to _process_X when interpretation is {RegressionInterpretation.RelativeToReference.value}")
      # X must be mutually exclusive
      if not is_boolean_dataframe_mutually_exclusive(X):
        raise RegressionInterpretationRelativeToReferenceMutualExclusivityRequirementsViolationException()
      X.drop(columns=[reference], inplace=True)
    elif interpretation == RegressionInterpretation.RelativeToBaseline:
      # X must not be mutually exclusive
      if is_boolean_dataframe_mutually_exclusive(X):
        raise RegressionInterpretationRelativeToBaselineMutualExclusivityRequirementsViolationException()
    elif interpretation == RegressionInterpretation.GrandMeanDeviation:
      # X must be mutually exclusive
      if not is_boolean_dataframe_mutually_exclusive(X):
        raise RegressionInterpretationGrandMeanDeviationMutualExclusivityRequirementsViolationException()
      X = one_hot_to_effect_coding(X, reference=reference)
    else:
      raise ValueError(f"\"{interpretation}\" is not a valid regression interpretation type.")

    X = X.astype(pd.Float32Dtype)
    if "const" in X.columns:
      raise ValueError(f"const is a reserved name. Please rename your subdataset.")
    if with_intercept:
      # Ignore coincidences where X has 1s already
      sm.add_constant(X, prepend=True, has_constant="add")
    return X