from typing import Sequence, cast
import numpy as np
from modules.api.wrapper import ApiResult
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.project.cache import ProjectCache
from modules.table.engine import TableEngine
from modules.table.filter_variants import AndTableFilter, NotEmptyTableFilter
from modules.topic.bertopic_ext.builder import EmptyBERTopicModelBuilder
from modules.topic.bertopic_ext.interpret import BERTopicInterpreter
from routes.table.model import CompareSubdatasetsSchema
from routes.table.model import TableTopicsResource

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

__all__ = [
  "compare_group_words"
]