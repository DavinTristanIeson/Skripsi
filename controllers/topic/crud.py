from typing import Sequence, cast

import numpy as np

from controllers.topic.dependency import _assert_dataframe_has_topic_columns
from models.topic import DocumentPerTopicResource, RefineTopicsSchema, TopicsOfColumnSchema
from modules.api.wrapper import ApiResult
from modules.config import TextualSchemaColumn
from modules.project.cache import ProjectCache
from modules.table import TableEngine, TablePaginationApiResult
from modules.table.filter_variants import EqualToTableFilter
from modules.table.pagination import PaginationParams
from modules.topic.bertopic_ext.builder import BERTopicModelBuilder, EmptyBERTopicModelBuilder
from modules.topic.bertopic_ext.interpret import BERTopicInterpreter
from modules.topic.model import Topic, TopicModelingResult


def paginate_documents_per_topic(cache: ProjectCache, column: TextualSchemaColumn, topic: Topic, params: PaginationParams)->TablePaginationApiResult[DocumentPerTopicResource]:
  df = cache.load_workspace()
  engine = TableEngine(cache.config)

  _assert_dataframe_has_topic_columns(df, column)

  params.filter = EqualToTableFilter(
    target=column.topic_column.name,
    value=topic.id,
  )
  params.sort = None
  
  filtered_df, meta = engine.paginate(df, params)

  documents: list[DocumentPerTopicResource] = []
  for idx, row in filtered_df.iterrows():
    document = DocumentPerTopicResource(
      id=int(idx), # type: ignore
      original=row[column.name],
      preprocessed=row[column.preprocess_column.name],
      topic=row[column.topic_column.name],
    )
    documents.append(document)

  return TablePaginationApiResult(
    data=documents,
    meta=meta,
    columns=cache.config.data_schema.columns,
    message=None,
  )

def refine_topics(cache: ProjectCache, body: RefineTopicsSchema, column: TextualSchemaColumn):
  df = cache.load_workspace()
  config = cache.config
  _assert_dataframe_has_topic_columns(df, column)

  from modules.topic.bertopic_ext import BERTopicInterpreter

  # Update document-topic mapping
  document_indices = list(map(lambda x: x.document_id, body.document_topics))
  new_topics = list(map(lambda x: x.topic_id, body.document_topics))
  df.loc[document_indices, column.topic_column.name] = new_topics

  documents = df[column.preprocess_column.name]
  document_topics = df[column.topic_column.name]
  mask = documents.notna()
  documents = documents[mask]
  document_topics = document_topics[mask]

  model_builder = EmptyBERTopicModelBuilder(
    column=column,
  )
  bertopic_model = model_builder.build()

  bertopic_model.fit(
    cast(list[str], documents),
    cast(np.ndarray, document_topics)
  )

  interpreter = BERTopicInterpreter(bertopic_model)
  new_topics = interpreter.extract_topics()

  topic_label_map = {topic.id: topic.label for topic in body.topics}

  for topic in new_topics:
    if topic.id in topic_label_map:
      topic.label = topic_label_map[topic.id]

  new_tm_result = TopicModelingResult.infer_from(
    project_id=config.project_id,
    document_topics=document_topics,
    topics=new_topics,
  )

  # Save model
  cache.save_bertopic(bertopic_model, column.name)
  cache.save_topic(new_tm_result, column.name)
  cache.save_workspace(df)

  return ApiResult(
    data=None,
    message="The topics have been successfully updated to your specifications.",
  )


def get_filtered_topics_of_column(cache: ProjectCache, body: TopicsOfColumnSchema, tm_result: TopicModelingResult, column: TextualSchemaColumn):
  df = cache.load_workspace()
  config = cache.config
  _assert_dataframe_has_topic_columns(df, column)
  bertopic_model = cache.load_bertopic(column.name)

  filtered_df = TableEngine(config).filter(df, body.filter)
  local_corpus = cast(Sequence[str], filtered_df[column.preprocess_column])

  interpreter = BERTopicInterpreter(bertopic_model)
  local_bow = interpreter.represent_as_bow(local_corpus)
  local_ctfidf = interpreter.represent_as_ctfidf(local_bow)
  local_ctfidf, unique_topics = interpreter.topic_ctfidfs_per_class(local_ctfidf, filtered_df[column.topic_column.name])

  topics: list[Topic] = []
  for idx, topic in enumerate(unique_topics):
    words = interpreter.get_weighted_words(local_ctfidf[idx])
    frequency = (filtered_df[column.topic_column.name] == idx).sum()
    topics.append(Topic(
      id=topic,
      frequency=frequency,
      words=words,
      label=interpreter.get_label(words),
    ))

  new_tm_result = TopicModelingResult.infer_from(config.project_id, filtered_df[column.topic_column.name], topics)

  return ApiResult(data=new_tm_result, message=None)


__all__ = [
  "paginate_documents_per_topic",
  "refine_topics"
]