import itertools
from modules.comparison import TableComparisonEngine
from modules.comparison.engine import StatisticTestResult
from modules.config import SchemaColumnTypeEnum
from modules.config.schema.base import ANALYZABLE_SCHEMA_COLUMN_TYPES
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache
from routes.table.controller.preprocess import TablePreprocessModule

from ..model import (
  PairwiseStatisticTestResultResource,
  PairwiseStatisticTestSchema,
  StatisticTestSchema,
  OmnibusStatisticTestSchema,
)

def statistic_test(params: StatisticTestSchema, cache: ProjectCache):
  config = cache.config
  config.data_schema.assert_of_type(params.column, ANALYZABLE_SCHEMA_COLUMN_TYPES)
  df = cache.workspaces.load()
  engine = TableComparisonEngine(
    config=config,
    groups=[params.group1, params.group2],
    exclude_overlapping_rows=params.exclude_overlapping_rows
  )
  result = engine.compare(
    df,
    column_name=params.column,
    statistic_test_preference=params.statistic_test_preference,
    effect_size_preference=params.effect_size_preference,
  )
  
  return result

def pairwise_statistic_test(cache: ProjectCache, input: PairwiseStatisticTestSchema):
  import scipy.stats
  preprocess = TablePreprocessModule(cache)
  column = preprocess.assert_column(input.column, supported_types=ANALYZABLE_SCHEMA_COLUMN_TYPES)
  workspace = cache.workspaces.load()

  results: list[StatisticTestResult] = []
  p_values: list[float] = []
  with ProvisionedLogger().disable(["TableEngine", "TableComparisonEngine"]):
    for group1, group2 in itertools.combinations(input.groups, 2):
      engine = TableComparisonEngine(
        config=cache.config,
        groups=[group1, group2],
        exclude_overlapping_rows=input.exclude_overlapping_rows,
      )
      comparison_result = engine.compare(
        df=workspace,
        column_name=input.column,
        statistic_test_preference=input.statistic_test_preference,
        effect_size_preference=input.effect_size_preference,
      )
      p_values.append(comparison_result.significance.p_value)
      results.append(comparison_result)

  adjusted_p_values = scipy.stats.false_discovery_control(p_values)
  for result, pvalue in zip(results, adjusted_p_values):
    result.significance.p_value = pvalue
  return PairwiseStatisticTestResultResource(
    column=column,
    results=results,
    groups=list(map(lambda group: group.name, input.groups)),
  )


def omnibus_statistic_test(cache: ProjectCache, input: OmnibusStatisticTestSchema):
  config = cache.config
  config.data_schema.assert_of_type(input.column, [
    SchemaColumnTypeEnum.Categorical,
    SchemaColumnTypeEnum.Continuous,
    SchemaColumnTypeEnum.OrderedCategorical,
    SchemaColumnTypeEnum.Temporal,
    SchemaColumnTypeEnum.Topic,
  ])
  df = cache.workspaces.load()
  engine = TableComparisonEngine(
    config=config,
    groups=input.groups,
    exclude_overlapping_rows=input.exclude_overlapping_rows
  )
  result = engine.compare_omnibus(
    df,
    column_name=input.column,
    statistic_test_preference=input.statistic_test_preference,
  )
  return result

__all__ = [
  "statistic_test",
  "pairwise_statistic_test",
  "omnibus_statistic_test"
]