
import numpy as np
import pandas as pd
from modules.comparison.base import EffectSizeResult, SignificanceResult
from modules.comparison.engine import TableComparisonEngine
from modules.comparison.statistic_test import StatisticTestMethodEnum
from modules.config.schema.base import CATEGORICAL_SCHEMA_COLUMN_TYPES
from modules.config.schema.schema_variants import SchemaColumn
from modules.project.cache import ProjectCache
from modules.table.filter_variants import NamedTableFilter
from routes.statistic_test.model import BinaryStatisticTestOnContingencyTableResultMainResource, BinaryStatisticTestOnContingencyTableResultResource, ContingencyTableResource, GetContingencyTableSchema
from routes.table.controller.preprocess import TablePreprocessModule

def __get_data_group_mapping(df: pd.DataFrame, column: SchemaColumn, groups: list[NamedTableFilter], preprocess: TablePreprocessModule):
  comparison_data = TableComparisonEngine(
    config=preprocess.cache.config,
    groups=groups
  ).extract_groups(df, column.name)

  full_data = preprocess.get_data(df, column)
  raw_group_mappings: list[pd.Series] = []
  for group in comparison_data.groups:
    raw_group_mappings.append(pd.Series(str(group.name), index=group.index))
  group_mappings = pd.concat(raw_group_mappings, axis=0)
  group_indices = group_mappings.index.intersection(full_data.index) # type: ignore

  group_data = full_data[group_indices]
  group_mappings = group_mappings[group_indices]
  return group_mappings, group_data

def contingency_table(cache: ProjectCache, input: GetContingencyTableSchema):
  preprocess = TablePreprocessModule(
    cache=cache
  )
  column = preprocess.assert_column(input.column, CATEGORICAL_SCHEMA_COLUMN_TYPES)
  df = cache.workspaces.load()
  grouping, data = __get_data_group_mapping(
    df=df,
    column=column,
    groups=input.groups,
    preprocess=preprocess,
  )

  observed = pd.crosstab(grouping, data)
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

def binary_statistic_test_on_contingency_table(cache: ProjectCache, input: GetContingencyTableSchema):
  import scipy.stats

  preprocess = TablePreprocessModule(
    cache=cache
  )
  column = preprocess.assert_column(input.column, CATEGORICAL_SCHEMA_COLUMN_TYPES)
  df = cache.workspaces.load()
  grouping, data = __get_data_group_mapping(
    df=df,
    column=column,
    preprocess=preprocess,
    groups=input.groups
  )
  global_contingency_table = pd.crosstab(grouping, data)
  global_contingency_table.fillna(0, inplace=True)

  results: list[list[BinaryStatisticTestOnContingencyTableResultResource]] = []
  for variable1 in global_contingency_table.index:
    results_row: list[BinaryStatisticTestOnContingencyTableResultResource] = []
    for variable2 in global_contingency_table.columns:
      TT = global_contingency_table.at[variable1, variable2]
      TF = global_contingency_table.loc[variable1, :].sum()
      FT = global_contingency_table.loc[:, variable2].sum()
      FF = global_contingency_table.sum().sum() - TT
      contingency_table = pd.DataFrame([
        [TT, TF],
        [FT, FF],
      ], index=[variable1, f"NOT {variable1}"], columns=[variable2, f"NOT {variable2}"])

      warnings = []
      has_zero_values = (contingency_table == 0).sum().sum() > 0
      has_less_than_five = (contingency_table < 5).sum().sum() > 0
      if has_zero_values:
        contingency_table += 0.5
      if has_less_than_five:
        warnings.append(f"There's a cell that contains a frequency less than 5 ({contingency_table.min().min()}). Results may not be accurate.")

      OR = (TT * FF) / (TF * FT)
      Q = (OR - 1) / (OR + 1)

      chisq_result = scipy.stats.chi2_contingency(contingency_table)

      results_row.append(BinaryStatisticTestOnContingencyTableResultResource(
        warnings=warnings,
        effect_size=EffectSizeResult(
          type="yule_q",
          value=Q,
        ),
        significance=SignificanceResult(
          type=StatisticTestMethodEnum.ChiSquared,
          statistic=chisq_result.statistic, # type: ignore
          p_value=chisq_result.pvalue # type: ignore
        ),
        frequency=TT,
        discriminator1=str(variable1),
        discriminator2=str(variable2),
      ))
    results.append(results_row)

  return BinaryStatisticTestOnContingencyTableResultMainResource(
    results=results,
    rows=list(map(str, global_contingency_table.index)),
    columns=list(map(str, global_contingency_table.columns)),
    column=column,
  )
  
__all__ = [
  "contingency_table",
  "binary_statistic_test_on_contingency_table"
]