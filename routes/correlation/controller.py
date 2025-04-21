from dataclasses import dataclass
from http import HTTPStatus
from typing import cast
import numpy as np
import pandas as pd
from modules.api.wrapper import ApiError
from modules.config.schema.base import CATEGORICAL_SCHEMA_COLUMN_TYPES, SchemaColumnTypeEnum
from modules.config.schema.schema_variants import SchemaColumn
from modules.project.cache import ProjectCache
from routes.correlation.model import ContingencyTableResource, TopicCorrelationSchema

@dataclass
class TableCorrelationPreprocessPartialResult:
  mask: pd.Series
  column: SchemaColumn

@dataclass
class TableCorrelationPreprocessModule:
  cache: ProjectCache

  def apply_partial(self, column_name: str, *, supported_types: list[SchemaColumnTypeEnum]):
    config = self.cache.config

    column = config.data_schema.assert_of_type(column_name, supported_types)
    df = self.cache.load_workspace()

    if column.name not in df.columns:
      raise ApiError(f"The column \"{column.name}\" does not exist in the dataset. There may have been some sort of data corruption in the application.", HTTPStatus.NOT_FOUND)
    
    mask = df[column.name].notna()
    if column.type == SchemaColumnTypeEnum.Topic:
      mask = mask & (df[column.name] != -1)

    return TableCorrelationPreprocessPartialResult(
      mask=mask,
      column=column
    )

  def consume(self, partial1: TableCorrelationPreprocessPartialResult, partial2: TableCorrelationPreprocessPartialResult):
    workspace = self.cache.load_workspace()
    mask = partial1.mask & partial2.mask
    return workspace.loc[mask, :]

  def extract(self, df: pd.DataFrame, column: SchemaColumn):
    if column.type == SchemaColumnTypeEnum.Topic:
      tm_result = self.cache.load_topic(cast(str, column.source_name))
      categorical_data = pd.Categorical(df[column.name])
      return cast(pd.Series, categorical_data.rename_categories(tm_result.renamer))
    return df[column.name]


def contingency_table(cache: ProjectCache, input: TopicCorrelationSchema):
  from scipy.cluster.hierarchy import linkage, leaves_list
  from scipy.spatial.distance import pdist, squareform

  preprocess = TableCorrelationPreprocessModule(cache=cache)
  supported_types = CATEGORICAL_SCHEMA_COLUMN_TYPES
  partial1 = preprocess.apply_partial(input.column1, supported_types=supported_types)
  partial2 = preprocess.apply_partial(input.column2, supported_types=supported_types)

  df = preprocess.consume(partial1, partial2)

  data1 = preprocess.extract(df, partial1.column)
  data2 = preprocess.extract(df, partial2.column)

  uniques1 = data1.unique()
  uniques2 = data2.unique()

  if len(uniques1) == 0:
    raise ApiError(
      f"It seems that {input.column1} doesn't contain any categories at all in the dataset so we cannot calculate the contingency table.",
      HTTPStatus.BAD_REQUEST
    )
  if len(uniques2) == 0:
    raise ApiError(
      f"It seems that {input.column2} doesn't contain any categories at all in the dataset so we cannot calculate the contingency table.",
      HTTPStatus.BAD_REQUEST
    )

  observed = pd.crosstab(data1, data2)
  observed.fillna(0, inplace=True)
  observed_npy = observed.to_numpy()
  # Calculate expected and residuals
  marginal_rows = observed_npy.sum(axis=1)
  marginal_cols = observed_npy.sum(axis=0)
  grand_total = observed_npy.sum()
  expected: np.ndarray = np.outer(marginal_rows, marginal_cols) / grand_total
  absolute_residuals: np.ndarray = observed_npy - expected
  standardized_residuals: np.ndarray = absolute_residuals / np.sqrt(expected)

   # hierarchical clustering
  if not partial1.column.is_ordered:
    # Compute linkage for rows
    row_linkage = linkage(pdist(absolute_residuals), method='average')
    row_order: np.ndarray = leaves_list(row_linkage)
  else:
    row_order = np.array(list(range(len(observed.index))))
  
  if not partial2.column.is_ordered:
    # Compute linkage for columns
    col_linkage = linkage(pdist(absolute_residuals.T), method='average')
    col_order: np.ndarray = leaves_list(col_linkage)
  else:
    col_order = np.array(list(range(len(observed.columns))))
 
  observed = observed.iloc[row_order, :].iloc[:, col_order]
  observed_npy = observed_npy[row_order, :][:, col_order]
  expected = expected[row_order, :][:, col_order]
  absolute_residuals = absolute_residuals[row_order, :][:, col_order]
  standardized_residuals = standardized_residuals[row_order, :][:, col_order]

  return ContingencyTableResource(
    rows=list(map(str, observed.index)),
    columns=list(map(str, observed.columns)),
    column1=partial1.column,
    column2=partial2.column,
    observed=observed_npy.tolist(),
    expected=expected.tolist(),
    residuals=absolute_residuals.tolist(),
    standardized_residuals=standardized_residuals.tolist(),
  )