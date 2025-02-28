from fastapi import APIRouter

from controllers.topic.crud import refine_topics
from modules.api.wrapper import ApiResult
from modules.table import PaginationParams

from controllers.project import ProjectCacheDependency
from controllers.topic import (
  TextualSchemaColumnDependency, TopicExistsDependency,
  TopicModelingResultDependency, paginate_documents_per_topic,
  check_topic_modeling_status, start_topic_modeling
)
from models.topic import RefineTopicsSchema, StartTopicModelingSchema


router = APIRouter(
  tags=["Topic Modeling"]
)

@router.post("/topic-modeling")
def post__start_topic_modeling(body: StartTopicModelingSchema, cache: ProjectCacheDependency):
  return start_topic_modeling(body, cache)

@router.get("/topic-modeling/status")
def get__topic_modeling__status(cache: ProjectCacheDependency, column: TextualSchemaColumnDependency):
  return check_topic_modeling_status(cache, column)

@router.get("/{column}/topics")
def get__topics(topic_modeling_result: TopicModelingResultDependency):
  return ApiResult(
    data=topic_modeling_result,
    message=None,
  )

@router.get("/{column}/topics/refine")
def refine__topics(body: RefineTopicsSchema, cache: ProjectCacheDependency, column: TextualSchemaColumnDependency, topic_modeling_result: TopicModelingResultDependency):
  return refine_topics(
    cache=cache,
    body=body,
    tm_result=topic_modeling_result,
    column=column,
  )

@router.get("/{column}/documents/{topic}")
def get__documents_per_topic(cache: ProjectCacheDependency, column: TextualSchemaColumnDependency, params: PaginationParams, topics: TopicExistsDependency):
  return paginate_documents_per_topic(
    cache=cache,
    column=column,
    params=params,
    topics=topics,
  )