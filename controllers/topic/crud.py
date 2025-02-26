from models.topic import DocumentPerTopicResource, RefineTopicsSchema
from modules.config import ProjectCache, TextualSchemaColumn
from modules.table import TableEngine, IsOneOfTableFilter, TableFilterTypeEnum, PaginatedApiResult
from modules.table.pagination import PaginationParams
from modules.topic.model import Topic


def paginate_documents_per_topic(cache: ProjectCache, column: TextualSchemaColumn, topics: list[Topic], params: PaginationParams)->PaginatedApiResult[DocumentPerTopicResource]:
  df = cache.load_workspace()
  engine = TableEngine(cache.config)
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

def refine_topics(cache: ProjectCache, body: RefineTopicsSchema, column: TextualSchemaColumn):
  df = cache.load_workspace()
  bertopic_model = cache.load_bertopic(column.name)

__all__ = [
  "paginate_documents_per_topic"
]