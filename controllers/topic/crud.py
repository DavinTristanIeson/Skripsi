import functools
from typing import Optional, Sequence, cast

import numpy as np
from models.topic import DocumentPerTopicResource, RefineTopicsSchema, TopicUpdateSchema
from modules.api.wrapper import ApiResult
from modules.baseclass import ValueCarrier
from modules.config import TextualSchemaColumn
from modules.project.cache import ProjectCache
from modules.table import TableEngine, IsOneOfTableFilter, TableFilterTypeEnum, TablePaginationApiResult
from modules.table.filter_variants import EqualToTableFilter
from modules.table.pagination import PaginationParams
from modules.topic.bertopic_ext.builder import BERTopicModelBuilder
from modules.topic.model import Topic, TopicModelingResult


def paginate_documents_per_topic(cache: ProjectCache, column: TextualSchemaColumn, topics: Optional[list[Topic]], params: PaginationParams)->TablePaginationApiResult[DocumentPerTopicResource]:
  df = cache.load_workspace()
  engine = TableEngine(cache.config)

  if topics is None:
    params.filter = EqualToTableFilter(
      target=column.name,
      type=TableFilterTypeEnum.EqualTo,
      value=-1,
    )
  else:
    topic_idx = list(map(lambda x: x.id, topics))
    params.filter = IsOneOfTableFilter(
      target=column.name,
      type=TableFilterTypeEnum.IsOneOf,
      values=list(map(str, topic_idx)),
    )
  params.sort = None
  
  filtered_df = engine.paginate(df, params)

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
    meta=engine.get_meta(documents, params),
    columns=cache.config.data_schema.columns,
    message=None,
  )


def refine_topics(cache: ProjectCache, body: RefineTopicsSchema, tm_result: TopicModelingResult, column: TextualSchemaColumn):
  import copy
  df = cache.load_workspace()
  config = cache.config

  from modules.topic.bertopic_ext import BERTopicInterpreter, BERTopicIndividualModels

  # Update document-topic mapping
  document_indices = list(map(lambda x: x.document_id, body.document_topics))
  new_topics = list(map(lambda x: x.topic_id, body.document_topics))
  df.loc[document_indices, column.topic_column.name] = new_topics

  documents = df[column.preprocess_column.name]
  document_topics = df[column.topic_column.name]
  mask = documents.notna()
  documents = documents[mask]
  document_topics = document_topics[mask]

  model_builder = BERTopicModelBuilder(
    project_id=config.project_id,
    column=column,
    corpus_size=len(documents)
  )

  bertopic_model = model_builder.build()

  bertopic_model.update_topics(
    docs=cast(list[str], documents),
    top_n_words=bertopic_model.top_n_words,
    vectorizer_model=model_builder.build_vectorizer_model(),
    ctfidf_model=model_builder.build_ctfidf_model(),
    topics=cast(list[int], document_topics)
  )

  interpreter = BERTopicInterpreter(bertopic_model)
  new_topics = interpreter.extract_topics()

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

__all__ = [
  "paginate_documents_per_topic",
  "refine_topics"
]