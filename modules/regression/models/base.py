import abc
from dataclasses import dataclass
from typing import cast

import pandas as pd

from modules.config.schema.base import ANALYZABLE_SCHEMA_COLUMN_TYPES, SchemaColumnTypeEnum
from modules.config.schema.schema_variants import SchemaColumn
from modules.project.cache import ProjectCache
from modules.regression.exceptions import NoIndependentVariableDataException
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
  
  def _load(self, groups: list[NamedTableFilter], target: str):
    cache = self.cache
    config = cache.config
    column = config.data_schema.assert_of_type(target, ANALYZABLE_SCHEMA_COLUMN_TYPES)

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
    
    # Use categorical dtype for topic
    Y = self._transform_data(Y, column)
    return X, Y
  
  def _process_X(self, X: pd.DataFrame, with_intercept: bool):
    pass