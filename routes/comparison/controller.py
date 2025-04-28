import http
from typing import Sequence, cast

import numpy as np
import pandas as pd
from modules.api.wrapper import ApiError, ApiResult
from modules.comparison import TableComparisonEngine
from modules.comparison.engine import TableComparisonEmptyException
from modules.config import SchemaColumnTypeEnum, TextualSchemaColumn
from modules.project.cache import ProjectCache
from modules.table import TableEngine, AndTableFilter, NotEmptyTableFilter
from modules.topic.bertopic_ext import BERTopicModelBuilder, BERTopicInterpreter

from .model import (
  ComparisonStatisticTestSchema,
  ComparisonGroupWordsSchema,
)
from routes.table.model import TableTopicsResource
from modules.topic.bertopic_ext.builder import EmptyBERTopicModelBuilder

def statistic_test(params: ComparisonStatisticTestSchema, cache: ProjectCache):
  config = cache.config
  config.data_schema.assert_of_type(params.column, [
    SchemaColumnTypeEnum.Categorical,
    SchemaColumnTypeEnum.Continuous,
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
  try:
    result = engine.compare(
      df,
      column_name=params.column,
      statistic_test_preference=params.statistic_test_preference,
      effect_size_preference=params.effect_size_preference,
    )
  except TableComparisonEmptyException as e:
    raise ApiError(e.args[0], http.HTTPStatus.BAD_REQUEST)
  
  return ApiResult(
    data=result,
    message=None
  )


def compare_group_words(params: ComparisonGroupWordsSchema, cache: ProjectCache):
  from bertopic import BERTopic

  config = cache.config
  column = cast(TextualSchemaColumn, config.data_schema.assert_of_type(params.column, [SchemaColumnTypeEnum.Textual]))
  df = cache.load_workspace()
  engine = TableEngine(config=config)

  builder = EmptyBERTopicModelBuilder(
    column=column,
  )
  bertopic_model = builder.build()

  documents: list[str] = []
  document_topics: list[int] = []
  for group_id, group in enumerate(params.groups):
    group_df = engine.filter(df, AndTableFilter(
      operands=[
        group.filter,
        NotEmptyTableFilter(
          target=column.preprocess_column.name,
        )
      ]
    ))
    subcorpus = cast(Sequence[str], group_df)
    documents.extend(group_df[column.preprocess_column.name])
    document_topics.extend([group_id] * len(subcorpus))

  bertopic_model.fit(
    cast(list[str], documents),
    y=np.array(document_topics)
  )
  interpreter = BERTopicInterpreter(bertopic_model)

  topics = interpreter.extract_topics(map_topics=True)

  return ApiResult(data=TableTopicsResource(
    column=column,
    topics=topics,
  ), message=None)
  

__all__ = [
  "statistic_test",
  "compare_group_words"
]