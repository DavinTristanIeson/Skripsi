from typing import Sequence, cast

import numpy as np
from modules.api.wrapper import ApiResult
from modules.comparison import TableComparisonEngine
from modules.config import SchemaColumnTypeEnum, TextualSchemaColumn
from modules.project.cache import ProjectCache
from modules.table import TableEngine, AndTableFilter, NotEmptyTableFilter
from modules.topic.bertopic_ext import BERTopicInterpreter

from .model import (
  ComparisonStatisticTestSchema,
  CompareSubdatasetsSchema,
  SubdatasetCooccurrenceResource,
)
from routes.table.model import TableTopicsResource
from modules.topic.bertopic_ext.builder import BERTopicModelBuilder, EmptyBERTopicModelBuilder

def statistic_test(params: ComparisonStatisticTestSchema, cache: ProjectCache):
  config = cache.config
  config.data_schema.assert_of_type(params.column, [
    SchemaColumnTypeEnum.Categorical,
    SchemaColumnTypeEnum.Continuous,
    SchemaColumnTypeEnum.OrderedCategorical,
    SchemaColumnTypeEnum.Temporal,
    SchemaColumnTypeEnum.Topic,
  ])
  df = cache.workspaces.load()
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


def compare_group_words(params: CompareSubdatasetsSchema, cache: ProjectCache):
  config = cache.config
  column = cast(TextualSchemaColumn, config.data_schema.assert_of_type(params.column, [SchemaColumnTypeEnum.Textual]))
  df = cache.workspaces.load()
  engine = TableEngine(config=config)

  documents: list[str] = []
  document_topics: list[int] = []
  for group_id, group in enumerate(params.groups):
    group_mask = engine.filter_mask(df, AndTableFilter(
      operands=[
        group.filter,
        NotEmptyTableFilter(
          target=column.preprocess_column.name,
        )
      ]
    ))
    group_df = df[group_mask]
    subcorpus = cast(Sequence[str], group_df[column.preprocess_column.name])
    documents.extend(subcorpus)
    document_topics.extend([group_id] * len(subcorpus))

  model_builder = EmptyBERTopicModelBuilder(
    column=column,
  )
  bertopic_model = model_builder.build()

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
  

def subdataset_cooccurrence(params: CompareSubdatasetsSchema, cache: ProjectCache):
  config = cache.config
  df = cache.workspaces.load()
  engine = TableEngine(config=config)

  masks = list(map(lambda group: engine.filter_mask(df, group.filter), params.groups))
  frequencies = list(map(lambda mask: mask.sum(), masks))
  group_names = list(map(lambda group: group.name, params.groups))
  cooccurrences = np.full((len(params.groups), len(params.groups)), 0)
  for gid1, group1 in enumerate(params.groups):
    mask1 = masks[gid1]
    for gid2, group2 in enumerate(params.groups):
      mask2 = masks[gid2]
      cooccur_mask = mask1 & mask2
      cooccurrence = cooccur_mask.sum()
      cooccurrences[gid1, gid2] += cooccurrence
      
  return SubdatasetCooccurrenceResource(
    labels=group_names,
    cooccurrences=cooccurrences.tolist(),
    frequencies=frequencies
  )

__all__ = [
  "statistic_test",
  "compare_group_words"
]