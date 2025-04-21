from dataclasses import dataclass
from http import HTTPStatus
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
    return TableCorrelationPreprocessPartialResult(
      mask=df[column.name].notna(),
      column=column
    )

  def consume(self, partial1: TableCorrelationPreprocessPartialResult, partial2: TableCorrelationPreprocessPartialResult):
    workspace = self.cache.load_workspace()
    mask = partial1.mask & partial2.mask
    return workspace.loc[mask, :]


def contingency_table(cache: ProjectCache, input: TopicCorrelationSchema):
  from scipy.cluster.hierarchy import linkage, leaves_list
  from scipy.spatial.distance import pdist, squareform

  preprocess = TableCorrelationPreprocessModule(cache=cache)
  supported_types = CATEGORICAL_SCHEMA_COLUMN_TYPES
  partial1 = preprocess.apply_partial(input.column1, supported_types=supported_types)
  partial2 = preprocess.apply_partial(input.column2, supported_types=supported_types)

  df = preprocess.consume(partial1, partial2)

  observed = pd.crosstab(df[input.column1], df[input.column2])
  observed.fillna(0, inplace=True)
  observed_npy = observed.to_numpy()

  # hierarchical clustering
  if not partial1.column.is_ordered:
    # Compute linkage for rows
    row_linkage = linkage(pdist(observed_npy), method='average')
    row_order: list[int] = leaves_list(row_linkage)
    observed = observed.iloc[row_order, :]
  
  if not partial2.column.is_ordered:
    # Compute linkage for columns
    col_linkage = linkage(pdist(observed_npy.T), method='average')
    col_order: list[int] = leaves_list(col_linkage)
    observed = observed.iloc[:, col_order]

  observed_npy = observed.to_numpy()
  # Calculate expected and residuals
  marginal_x = observed_npy.sum(axis=0)
  marginal_y = observed_npy.sum(axis=1)
  grand_total = observed_npy.sum()
  expected: np.ndarray = np.outer(marginal_x, marginal_y) / grand_total
  absolute_residuals: np.ndarray = observed_npy - expected
  pearson_residuals: np.ndarray = absolute_residuals / np.sqrt(expected)

  return ContingencyTableResource(
    rows=list(map(str, observed.index)),
    columns=list(map(str, observed.columns)),
    observed=observed_npy.tolist(),
    expected=expected.tolist(),
    residuals=absolute_residuals.tolist(),
    standardized_residuals=pearson_residuals.tolist(),
  )