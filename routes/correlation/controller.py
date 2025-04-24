from dataclasses import dataclass
from http import HTTPStatus
from typing import cast
import numpy as np
import pandas as pd
from modules.api.wrapper import ApiError
from modules.comparison.base import EffectSizeResult, SignificanceResult
from modules.comparison.engine import TableComparisonEngine
from modules.comparison.statistic_test import StatisticTestMethodEnum
from modules.config.schema.base import ANALYZABLE_SCHEMA_COLUMN_TYPES, CATEGORICAL_SCHEMA_COLUMN_TYPES, SchemaColumnTypeEnum
from modules.config.schema.schema_variants import SchemaColumn
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache
from modules.table.engine import TableEngine
from modules.table.filter_variants import EqualToTableFilter, NotTableFilter
from routes.correlation.model import BinaryStatisticTestOnContingencyTableMainResource, BinaryStatisticTestOnContingencyTableResource, BinaryStatisticTestOnDistributionResource, BinaryStatisticTestSchema, ContingencyTableResource, TopicCorrelationSchema

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
    df = workspace.loc[mask, :]
        
    if len(df) == 0:
      raise ApiError("There are no rows that can be visualized. Perhaps the filter is too strict; try adjusting the filter to be more lax.", HTTPStatus.BAD_REQUEST)

    sort_by = []
    if partial1.column.is_ordered:
      sort_by.append(partial1.column.name)
    if partial2.column.is_ordered:
      sort_by.append(partial2.column.name)

    df.sort_values(by=sort_by, inplace=True)
    return df 

  def extract(self, df: pd.DataFrame, column: SchemaColumn, *, transform_topics: bool = True):
    if column.type == SchemaColumnTypeEnum.Topic and transform_topics:
      tm_result = self.cache.load_topic(cast(str, column.source_name))
      categorical_data = pd.Categorical(df[column.name])
      return cast(pd.Series, categorical_data.rename_categories(tm_result.renamer))
    
    data = df[column.name]
    uniques = data.unique()
    if len(uniques) == 0:
      raise ApiError(
        f"It seems that {column.name} doesn't contain any values at all in the dataset so we cannot proceed with this operation.",
        HTTPStatus.BAD_REQUEST
      )
    return data
  
  
  def label_binary_variable(self, binary_variables: pd.Series | np.ndarray, column: SchemaColumn):
    if column.type == SchemaColumnTypeEnum.Topic:
      tm_result = self.cache.load_topic(cast(str, column.source_name))
      get_topics = map(lambda var: tm_result.find(var), binary_variables)
      topic_labels = map(lambda topic: topic.default_label if topic else None, get_topics)
      topic_default_labels = map(lambda label, value: label or str(value), topic_labels, binary_variables)
      return list(topic_default_labels)
    else:
      return list(map(str, binary_variables))

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

def binary_statistic_test_on_distribution(cache: ProjectCache, input: BinaryStatisticTestSchema):
  preprocess = TableCorrelationPreprocessModule(cache=cache)
  partial1 = preprocess.apply_partial(input.column1, supported_types=CATEGORICAL_SCHEMA_COLUMN_TYPES)
  partial2 = preprocess.apply_partial(input.column2, supported_types=ANALYZABLE_SCHEMA_COLUMN_TYPES)

  workspace = cache.load_workspace()
  total_count = len(workspace)

  df = preprocess.consume(partial1, partial2)

  discriminator = preprocess.extract(df, partial1.column, transform_topics=False)
  binary_variables = discriminator.unique()
  binary_variable_labels = preprocess.label_binary_variable(binary_variables, partial1.column)

  results: list[BinaryStatisticTestOnDistributionResource] = []
  with ProvisionedLogger().disable(["TableEngine", "TableComparisonEngine"]):
    for variable, label in zip(binary_variables, binary_variable_labels):
      filter = EqualToTableFilter(target=input.column1, value=variable)
      anti_filter = NotTableFilter(operand=filter)
      treatment_group = NamedTableFilter(name=label, filter=filter)
      control_group = NamedTableFilter(name=label, filter=anti_filter)
      engine = TableComparisonEngine(
        config=cache.config,
        engine=TableEngine(config=cache.config),
        groups=[treatment_group, control_group],
        exclude_overlapping_rows=True,
      )
      result = engine.compare(
        df,
        column_name=input.column2,
        statistic_test_preference=input.statistic_test_preference,
        effect_size_preference=input.effect_size_preference,
      )

      yes_count = result.groups[0].valid_count
      no_count = result.groups[1].valid_count
      results.append(BinaryStatisticTestOnDistributionResource(
        warnings=result.warnings,
        effect_size=result.effect_size,
        significance=result.significance,
        yes_count=yes_count,
        no_count=no_count,
        invalid_count=total_count - yes_count - no_count,
        discriminator=label,
      ))
  return results
  
def binary_statistic_test_on_contingency_table(cache: ProjectCache, input: BinaryStatisticTestSchema):
  import scipy.stats
  preprocess = TableCorrelationPreprocessModule(cache=cache)
  partial1 = preprocess.apply_partial(input.column1, supported_types=CATEGORICAL_SCHEMA_COLUMN_TYPES)
  partial2 = preprocess.apply_partial(input.column2, supported_types=ANALYZABLE_SCHEMA_COLUMN_TYPES)

  df = preprocess.consume(partial1, partial2)

  discriminator1 = preprocess.extract(df, partial1.column, transform_topics=False)
  discriminator2 = preprocess.extract(df, partial2.column, transform_topics=False)

  binary_variables1 = discriminator1.unique()
  binary_variables2 = discriminator2.unique()

  binary_variable_names1 = preprocess.label_binary_variable(binary_variables1, partial1.column)
  binary_variable_names2 = preprocess.label_binary_variable(binary_variables2, partial2.column)

  global_contingency_table = pd.crosstab(discriminator1, discriminator2)
  results: list[list[BinaryStatisticTestOnContingencyTableResource]] = []
  for variable1, label1 in zip(binary_variables1, binary_variable_names1):
    results_row: list[BinaryStatisticTestOnContingencyTableResource] = []
    for variable2, label2 in zip(binary_variables2, binary_variable_names2):
      TT = global_contingency_table.at[variable1, variable2]
      TF = global_contingency_table.loc[variable1, :].sum()
      FT = global_contingency_table.loc[:, variable2].sum()
      FF = global_contingency_table.sum().sum() - TT
      contingency_table = pd.DataFrame([
        [TT, TF],
        [FT, FF],
      ], index=[label1, f"NOT {label1}"], columns=[label2, f"NOT {label2}"])

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

      results_row.append(BinaryStatisticTestOnContingencyTableResource(
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
        discriminator1=label1,
        discriminator2=label2,
      ))
    results.append(results_row)

  return BinaryStatisticTestOnContingencyTableMainResource(
    results=results,
    rows=binary_variable_names1,
    columns=binary_variable_names2,
    column1=partial1.column,
    column2=partial2.column,
  )
  