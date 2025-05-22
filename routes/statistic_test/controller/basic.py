from modules.comparison import TableComparisonEngine
from modules.config import SchemaColumnTypeEnum
from modules.config.schema.base import ANALYZABLE_SCHEMA_COLUMN_TYPES
from modules.project.cache import ProjectCache

from ..model import (
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
]