from typing import Sequence, cast
from modules.api.wrapper import ApiResult
from modules.comparison import TableComparisonEngine
from modules.config import ProjectCache
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.table import TableEngine

from models.table import ComparisonStatisticTestSchema, ComparisonGroupWordsSchema, TableTopicsResource
from modules.topic.model import Topic

def statistic_test(params: ComparisonStatisticTestSchema, cache: ProjectCache):
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


def compare_group_words(params: ComparisonGroupWordsSchema, cache: ProjectCache):
  from modules.topic import BERTopicInterpreter
  config = cache.config
  column = config.data_schema.assert_of_type(params.column, [SchemaColumnTypeEnum.Textual])
  df = cache.load_workspace()
  engine = TableEngine(config=config)

  bertopic_model = cache.load_bertopic(column.name)
  interpreter = BERTopicInterpreter(bertopic_model)

  topics: list[Topic] = []
  for group_id, group in enumerate(params.groups):
    group_df = engine.filter(df, group.filter)
    subcorpus = cast(Sequence[str], group_df)
    bow = interpreter.represent_as_bow(subcorpus)
    ctfidf = interpreter.represent_as_ctfidf(bow)
    group_words = interpreter.get_weighted_words(ctfidf)
    group_label = interpreter.get_label(ctfidf) or f"Topic {group_id}"

    topics.append(Topic(
      id=group_id,
      frequency=len(group_df),
      label=group_label,
      words=group_words,
    ))
  
  return TableTopicsResource(
    column=column,
    topics=topics,
  )
  

__all__ = [
  "statistic_test",
  "compare_group_words"
]