
import numpy as np
import pandas as pd
from modules.comparison.engine import TableComparisonEngine
from modules.config.schema.base import CATEGORICAL_SCHEMA_COLUMN_TYPES
from modules.project.cache import ProjectCache
from modules.table.engine import TableEngine
from routes.statistical_analysis.model import ContingencyTableResource, GetContingencyTableSchema, GetSubdatasetCooccurrenceSchema, SubdatasetCooccurrenceResource
from routes.table.controller.preprocess import TablePreprocessModule

def contingency_table(cache: ProjectCache, input: GetContingencyTableSchema):
  preprocess = TablePreprocessModule(
    cache=cache
  )
  column = preprocess.assert_column(input.column, CATEGORICAL_SCHEMA_COLUMN_TYPES)
  df = preprocess.load_dataframe(filter=None)
  __data, df = preprocess.get_data(df, column, exclude_invalid=True, transform_data=True)

  comparison_data = TableComparisonEngine(
    config=preprocess.cache.config,
    groups=input.groups,
    exclude_overlapping_rows=input.exclude_overlapping_rows,
  ).extract_groups(df, column.name)

  comparison_data.groups = list(map(
    lambda group: preprocess.transform_data(group, column),
    comparison_data.groups
  ))

  frequency_distributions = [pd.Series(group.value_counts(), name=group.name) for group in comparison_data.groups]
  observed = pd.concat(frequency_distributions, axis=1).T
  observed.fillna(0, inplace=True)

  observed_npy = observed.to_numpy()
  # Calculate expected and residuals
  marginal_rows = observed_npy.sum(axis=1)
  marginal_cols = observed_npy.sum(axis=0)
  grand_total = observed_npy.sum()
  expected: np.ndarray = np.outer(marginal_rows, marginal_cols) / grand_total
  absolute_residuals: np.ndarray = observed_npy - expected
  standardized_residuals: np.ndarray = absolute_residuals / np.sqrt(expected)

  return ContingencyTableResource(
    rows=list(map(str, observed.index)),
    columns=list(map(str, observed.columns)),
    column=column,
    observed=observed_npy.tolist(),
    expected=expected.tolist(),
    residuals=absolute_residuals.tolist(),
    standardized_residuals=standardized_residuals.tolist(),
  )


def subdataset_cooccurrence(params: GetSubdatasetCooccurrenceSchema, cache: ProjectCache):
  config = cache.config
  df = cache.workspaces.load()
  engine = TableEngine(config=config)

  masks = list(map(lambda group: engine.filter_mask(df, group.filter), params.groups))
  frequencies = list(map(lambda mask: mask.sum(), masks))
  group_names = list(map(lambda group: group.name, params.groups))
  cooccurrences = np.full((len(params.groups), len(params.groups)), 0)
  correlations = np.full((len(params.groups), len(params.groups)), 0.0, dtype=np.float32)
  for gid1, group1 in enumerate(params.groups):
    mask1 = masks[gid1]
    for gid2, group2 in enumerate(params.groups):
      mask2 = masks[gid2]
      cooccur_mask = mask1 & mask2
      cooccurrence = cooccur_mask.sum()
      cooccurrences[gid1, gid2] += cooccurrence
      # Pearson correlation reduces to point biserial correlation for 0 and 1.
      correlation_matrix = np.corrcoef(
        mask1.to_numpy(dtype=np.float32),
        mask2.to_numpy(dtype=np.float32)
      )
      correlations[gid1, gid2] = correlation_matrix[0, 1]
      
  return SubdatasetCooccurrenceResource(
    labels=group_names,
    cooccurrences=cooccurrences.tolist(),
    correlations=correlations.tolist(),
    frequencies=frequencies
  )

  
__all__ = [
  "contingency_table",
  "subdataset_cooccurrence"
]