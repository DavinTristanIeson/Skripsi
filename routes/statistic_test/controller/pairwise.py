import itertools
from modules.comparison.engine import TableComparisonEngine, StatisticTestResult
from modules.config.schema.base import ANALYZABLE_SCHEMA_COLUMN_TYPES
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache
from modules.table.filter_variants import NamedTableFilter, NotTableFilter
from ..model import BinaryStatisticTestOnDistributionResultResource, BinaryStatisticTestSchema, PairwiseStatisticTestResultResource, PairwiseStatisticTestSchema
from routes.table.controller.preprocess import TablePreprocessModule

def pairwise_statistic_test(cache: ProjectCache, input: PairwiseStatisticTestSchema):
  import scipy.stats
  preprocess = TablePreprocessModule(cache)
  column = preprocess.assert_column(input.column, supported_types=ANALYZABLE_SCHEMA_COLUMN_TYPES)
  workspace = cache.workspaces.load()

  results: list[StatisticTestResult] = []
  p_values: list[float] = []
  with ProvisionedLogger().disable(["TableEngine", "TableComparisonEngine"]):
    for group1, group2 in itertools.combinations(input.groups, 2):
      engine = TableComparisonEngine(config=cache.config, groups=[group1, group2])
      comparison_result = engine.compare(
        df=workspace,
        column_name=input.column,
        statistic_test_preference=input.statistic_test_preference,
        effect_size_preference=input.effect_size_preference,
      )
      p_values.append(comparison_result.significance.p_value)
      results.append(comparison_result)

  adjusted_p_values = scipy.stats.false_discovery_control(p_values)
  return PairwiseStatisticTestResultResource(
    column=column,
    results=results,
    adjusted_p_values=adjusted_p_values.tolist(),
    groups=list(map(lambda group: group.name, input.groups)),
  )


def binary_statistic_test_on_distribution(cache: ProjectCache, input: BinaryStatisticTestSchema):
  preprocess = TablePreprocessModule(cache)
  column = preprocess.assert_column(input.column, supported_types=ANALYZABLE_SCHEMA_COLUMN_TYPES)
  workspace = cache.workspaces.load()

  results: list[StatisticTestResult] = []
  with ProvisionedLogger().disable(["TableEngine", "TableComparisonEngine"]):
    for group in input.groups:
      anti_group = NamedTableFilter(
        name=f"NOT {group.name}",
        filter=NotTableFilter(operand=group.filter)
      )
      engine = TableComparisonEngine(config=cache.config, groups=[group, anti_group])
      result = engine.compare(
        workspace,
        column_name=input.column,
        statistic_test_preference=input.statistic_test_preference,
        effect_size_preference=input.effect_size_preference,
      )
      results.append(result)
  return BinaryStatisticTestOnDistributionResultResource(
    column=column,
    groups=list(map(lambda group: group.name, input.groups)),
    results=results,
  )


__all__ = [
  "pairwise_statistic_test",
  "binary_statistic_test_on_distribution"
]