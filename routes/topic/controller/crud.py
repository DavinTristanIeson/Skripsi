from typing import Sequence, cast

import numpy as np
import pandas as pd

from modules.table.serialization import serialize_pandas
from modules.task.storage import TaskStorage
from routes.topic.model import DocumentPerTopicResource, RefineTopicsSchema, TopicsOfColumnSchema
from modules.api.wrapper import ApiResult
from modules.config import TextualSchemaColumn
from modules.project.cache import ProjectCache
from modules.table import TableEngine, TablePaginationApiResult
from modules.table.filter_variants import AndTableFilter, NotEmptyTableFilter
from modules.table.pagination import PaginationParams
from modules.topic.bertopic_ext.builder import EmptyBERTopicModelBuilder
from modules.topic.bertopic_ext.interpret import BERTopicInterpreter
from modules.topic.model import Topic, TopicModelingResult


def paginate_documents_per_topic(cache: ProjectCache, column: TextualSchemaColumn, params: PaginationParams)->TablePaginationApiResult[DocumentPerTopicResource]:
  df = cache.workspaces.load()
  engine = TableEngine(cache.config)

  column.assert_internal_columns(df, with_preprocess=True, with_topics=True)
  
  not_empty_filter = NotEmptyTableFilter(target=column.preprocess_column.name)
  if params.filter is not None:
    params.filter = AndTableFilter(
      operands=[
        not_empty_filter,
        params.filter,
      ]
    )
  else:
    params.filter = not_empty_filter

  filtered_df, meta = engine.paginate_workspace(params)

  documents: list[DocumentPerTopicResource] = []
  for row in serialize_pandas(filtered_df):
    topic_id = row[column.topic_column.name]
    document = DocumentPerTopicResource(
      id=int(row["__index"]),
      original=row[column.name],
      preprocessed=row[column.preprocess_column.name],
      topic=None if pd.isna(topic_id) else topic_id,
    )
    documents.append(document)

  return TablePaginationApiResult(
    data=documents,
    meta=meta,
    columns=cache.config.data_schema.columns,
    message=None,
  )

def refine_topics(cache: ProjectCache, body: RefineTopicsSchema, column: TextualSchemaColumn):
  df = cache.workspaces.load()
  config = cache.config
  column.assert_internal_columns(df, with_preprocess=True, with_topics=True)

  from modules.topic.bertopic_ext import BERTopicInterpreter

  # Update document-topic mapping
  document_indices = list(map(lambda x: x.document_id, body.document_topics))
  new_topics = list(map(lambda x: x.topic_id, body.document_topics))
  df.loc[document_indices, column.topic_column.name] = new_topics
  df[column.topic_column.name] = df[column.topic_column.name].astype("Int32")

  documents = df[column.preprocess_column.name]
  document_topics = df[column.topic_column.name]
  mask = documents.notna()
  documents = documents[mask]
  document_topics = document_topics[mask]

  # Reset BERTopic model
  model_builder = EmptyBERTopicModelBuilder(
    column=column,
  )
  bertopic_model = model_builder.build()

  bertopic_model.fit(
    cast(list[str], documents),
    y=cast(np.ndarray, document_topics)
  )

  interpreter = BERTopicInterpreter(bertopic_model)
  new_topics = interpreter.extract_topics()

  # Assign new labels
  topic_map = {topic.id: topic for topic in body.topics}
  for topic in new_topics:
    if topic.id in topic_map:
      topic.label = topic_map[topic.id].label
      topic.description = topic_map[topic.id].description
      topic.tags = topic_map[topic.id].tags

  new_tm_result = TopicModelingResult.infer_from(
    project_id=config.project_id,
    document_topics=document_topics,
    topics=new_topics,
  )

  # Save model
  cache.bertopic_models.save(bertopic_model, column.name)
  cache.topics.save(new_tm_result, column.name)
  cache.workspaces.save(df)
  # Invalidate tasks
  TaskStorage().invalidate(prefix=config.project_id)
  # Clean up experiments
  config.paths.cleanup_topic_experiments()

  return ApiResult(
    data=None,
    message="The topics have been successfully updated to your specifications.",
  )


def get_filtered_topics_of_column(cache: ProjectCache, body: TopicsOfColumnSchema, column: TextualSchemaColumn, tm_result: TopicModelingResult):
  df = cache.workspaces.load()
  config = cache.config
  column.assert_internal_columns(df, with_preprocess=True, with_topics=True)
  bertopic_model = cache.bertopic_models.load(column.name)

  filtered_df = TableEngine(config).filter(df, body.filter)
  local_corpus = cast(Sequence[str], filtered_df[column.preprocess_column])

  interpreter = BERTopicInterpreter(bertopic_model)
  local_bow = interpreter.represent_as_bow_sparse(local_corpus)
  local_ctfidf = interpreter.represent_as_ctfidf(local_bow)
  tuned_ctfidf, unique_topics = interpreter.topic_ctfidfs_per_class(local_ctfidf, filtered_df[column.topic_column.name])

  topics: list[Topic] = []
  for idx, topic in enumerate(unique_topics):
    existing_topic = tm_result.find(topic)
    if not existing_topic:
      continue

    tuned_ctfidf_nparray = tuned_ctfidf.getrow(idx).toarray()[0]
    words = interpreter.get_weighted_words(tuned_ctfidf_nparray)
    topics.append(Topic(
      id=topic,
      frequency=existing_topic.frequency,
      words=words,
      label=existing_topic.label
    ))

  new_tm_result = TopicModelingResult.infer_from(config.project_id, filtered_df[column.topic_column.name], topics)

  return ApiResult(data=new_tm_result, message=None)


__all__ = [
  "paginate_documents_per_topic",
  "refine_topics",
  "get_filtered_topics_of_column"
]