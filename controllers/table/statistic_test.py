from modules.api.wrapper import ApiResult
from modules.comparison import TableComparisonEngine
from modules.config import ProjectCache
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.table import TableEngine

from models.table import TableColumnsStatisticTestSchema

def statistic_test(params: TableColumnsStatisticTestSchema, cache: ProjectCache):
  config = cache.config
  config.data_schema.assert_of_type(params.column, [
    SchemaColumnTypeEnum.Categorical,
    SchemaColumnTypeEnum.Continuous,
    SchemaColumnTypeEnum.MultiCategorical,
    SchemaColumnTypeEnum.OrderedCategorical,
    SchemaColumnTypeEnum.Temporal,
    SchemaColumnTypeEnum.Topic,
  ])
  df = cache.load_workspace()
  engine = TableComparisonEngine(
    config=config,
    engine=TableEngine(config),
    groups=[params.group1, params.group2],
    exclude_overlapping_rows=params.exclude_overlapping_rows
  )
  result = engine.compare(
    df,
    column_name=params.column,
    statistic_test_preference=params.statistic_test_preference,
    effect_size_preference=params.effect_size_preference,
  )
  return ApiResult(
    data=result,
    message=None
  )

__all__ = [
  "statistic_test"
]