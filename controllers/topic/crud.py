from typing import Optional, Sequence, cast
from models.topic import DocumentPerTopicResource, RefineTopicsSchema
from modules.config import ProjectCache, TextualSchemaColumn
from modules.table import TableEngine, IsOneOfTableFilter, TableFilterTypeEnum, PaginatedApiResult
from modules.table.filter_variants import EqualToTableFilter
from modules.table.pagination import PaginationParams
from modules.topic.model import Topic, TopicModelingResult


def paginate_documents_per_topic(cache: ProjectCache, column: TextualSchemaColumn, topics: Optional[list[Topic]], params: PaginationParams)->PaginatedApiResult[DocumentPerTopicResource]:
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

  return PaginatedApiResult(
    data=documents,
    meta=engine.get_meta(df, params),
    message=None,
  )

def refine_topics(cache: ProjectCache, body: RefineTopicsSchema, tm_result: TopicModelingResult, column: TextualSchemaColumn):
  df = cache.load_workspace()
  config = cache.config
  bertopic_model = cache.load_bertopic(column.name)

  from modules.topic.bertopic_ext import BERTopicInterpreter, BERTopicIndividualModels

  # Update document-topic mapping
  document_indices = list(map(lambda x: x.document_id, body.document_topics))
  new_topics = list(map(lambda x: x.topic_id, body.document_topics))
  df.loc[document_indices, column.topic_column.name] = new_topics

  documents = df[column.preprocess_column.name]
  documents = cast(list[str], documents[documents.notna()])

  individual_models = BERTopicIndividualModels.cast(bertopic_model)
  bertopic_model.update_topics(
    documents,
    top_n_words=bertopic_model.top_n_words,
    vectorizer_model=individual_models.vectorizer_model,
    ctfidf_model=individual_models.ctfidf_model
  )
  cache.bertopic_models.invalidate(key=column.name)

  interpreter = BERTopicInterpreter(bertopic_model)
  topics = interpreter.extract_topics()

  # Resolve differences
  topic_label_mapping = {topic.id: topic.label for topic in body.topics}
  for topic in topics:
    topic.label = topic_label_mapping.get(topic.id, topic.label)

  # Resolve hierarchy updates
  for topic in topics:
    pass

  # Update topic modeling result
  
  tm_result.hierarchy = body.hierarchy
  tm_result.reindex()



__all__ = [
  "paginate_documents_per_topic"
]