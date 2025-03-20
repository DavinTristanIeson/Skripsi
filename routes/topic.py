from fastapi import APIRouter

from modules.api.wrapper import ApiError, ApiResult
from modules.table import PaginationParams
from modules.table.pagination import TablePaginationApiResult
from modules.task.responses import TaskResponse
from modules.topic.model import TopicModelingResult

from controllers.topic.crud import refine_topics
from controllers.project import ProjectCacheDependency
from controllers.topic import (
  TextualSchemaColumnDependency, TopicExistsDependency,
  TopicModelingResultDependency, paginate_documents_per_topic,
  check_topic_modeling_status, start_topic_modeling
)
from models.topic import ColumnTopicModelingResultResource, DocumentPerTopicResource, RefineTopicsSchema, StartTopicModelingSchema


router = APIRouter(
  tags=["Topic Modeling"]
)

@router.post("/start")
def post__start_topic_modeling(body: StartTopicModelingSchema, cache: ProjectCacheDependency, column: TextualSchemaColumnDependency)->ApiResult[None]:
  return start_topic_modeling(body, cache, column)

@router.get("/status")
def get__topic_modeling__status(cache: ProjectCacheDependency, column: TextualSchemaColumnDependency)->TaskResponse[TopicModelingResult]:
  return check_topic_modeling_status(cache, column)

@router.get("/")
def get__all_topics(cache: ProjectCacheDependency)->ApiResult[list[ColumnTopicModelingResultResource]]:
  config = cache.config
  textual_columns = config.data_schema.textual()
  topic_modeling_results: list[ColumnTopicModelingResultResource] = []
  for column in textual_columns:
    try:
      result = cache.load_topic(column.name)
    except ApiError:
      result = None
    topic_modeling_results.append(ColumnTopicModelingResultResource(
      result=result,
      column=column,
    ))
  return ApiResult(
    data=topic_modeling_results,
    message=None,
  )

@router.put("/refine")
def refine__topics(
  body: RefineTopicsSchema,
  cache: ProjectCacheDependency,
  column: TextualSchemaColumnDependency,
  topic_modeling_result: TopicModelingResultDependency
)->ApiResult[None]:
  return refine_topics(
    cache=cache,
    body=body,
    tm_result=topic_modeling_result,
    column=column,
  )

@router.get("/documents")
def get__documents_per_topic(cache: ProjectCacheDependency, column: TextualSchemaColumnDependency, params: PaginationParams, topics: TopicExistsDependency)->TablePaginationApiResult[DocumentPerTopicResource]:
  return paginate_documents_per_topic(
    cache=cache,
    column=column,
    params=params,
    topics=topics,
  )