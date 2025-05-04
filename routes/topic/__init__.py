from fastapi import APIRouter

from controllers.project import ProjectCacheDependency
from controllers.topic import TextualSchemaColumnDependency, TopicModelingResultDependency
from modules.api.wrapper import ApiError, ApiResult
from modules.table import PaginationParams
from modules.table.pagination import TablePaginationApiResult
from modules.task.responses import TaskResponse
from modules.topic.model import TopicModelingResult
from routes.topic.controller.evaluation import check_topic_model_experiment_status, perform_topic_model_evaluation, perform_topic_model_experiment

from .controller import (
  get_document_visualization_results, get_topic_visualization_results,
  get_filtered_topics_of_column, refine_topics,
  paginate_documents_per_topic,
  check_topic_modeling_status, start_topic_modeling
)
from .model import (
  BERTopicExperimentTaskRequest, ColumnTopicModelingResultResource, DocumentPerTopicResource,
  DocumentTopicsVisualizationResource, EvaluateTopicModelResultTaskRequest, RefineTopicsSchema,
  StartTopicModelingSchema, TopicModelExperimentSchema, TopicVisualizationResource, TopicsOfColumnSchema
)


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
def get__all_topic_modeling_results(cache: ProjectCacheDependency)->ApiResult[list[ColumnTopicModelingResultResource]]:
  config = cache.config
  textual_columns = config.data_schema.textual()
  topic_modeling_results: list[ColumnTopicModelingResultResource] = []
  for column in textual_columns:
    try:
      result = cache.topics.load(column.name)
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

@router.post("/topics")
def get__all_topics(cache: ProjectCacheDependency, column: TextualSchemaColumnDependency, tm_result: TopicModelingResultDependency, body: TopicsOfColumnSchema)->ApiResult[TopicModelingResult]:
  if body.filter is None:
    return ApiResult(data=tm_result, message=None)
  return get_filtered_topics_of_column(cache, body, column, tm_result)

@router.put("/refine")
def put__refine_topics(
  body: RefineTopicsSchema,
  cache: ProjectCacheDependency,
  column: TextualSchemaColumnDependency,
  topic_modeling_result: TopicModelingResultDependency
)->ApiResult[None]:
  return refine_topics(
    cache=cache,
    body=body,
    column=column,
  )

@router.post("/documents")
def post__documents_per_topic(
  cache: ProjectCacheDependency,
  column: TextualSchemaColumnDependency,
  params: PaginationParams,
  topic_modeling_result: TopicModelingResultDependency
)->TablePaginationApiResult[DocumentPerTopicResource]:
  return paginate_documents_per_topic(
    cache=cache,
    column=column,
    params=params,
  )

@router.get("/visualization/topics")
def get__topic_visualization_results(
  cache: ProjectCacheDependency,
  column: TextualSchemaColumnDependency,
  topic_modeling_result: TopicModelingResultDependency
)->ApiResult[list[TopicVisualizationResource]]:
  return get_topic_visualization_results(
    cache=cache,
    column=column,
    topic_modeling_result=topic_modeling_result
  )

@router.get("/visualization/documents")
def get__document_visualization_results(
  cache: ProjectCacheDependency,
  column: TextualSchemaColumnDependency,
  topic_modeling_result: TopicModelingResultDependency
)->ApiResult[DocumentTopicsVisualizationResource]:
  return get_document_visualization_results(
    cache=cache,
    column=column,
    topic_modeling_result=topic_modeling_result
  )

@router.post('/evaluation/start')
def post__start_topic_evaluation(
  cache: ProjectCacheDependency,
  column: TextualSchemaColumnDependency,
  topic_modeling_result: TopicModelingResultDependency,
):
  return perform_topic_model_evaluation(EvaluateTopicModelResultTaskRequest(
    project_id=cache.config.project_id,
    column=column.name,
  ))

@router.get('/evaluation/status')
def get__topic_evaluation_status(
  cache: ProjectCacheDependency,
  column: TextualSchemaColumnDependency,
  topic_modeling_result: TopicModelingResultDependency,
):
  return perform_topic_model_evaluation(EvaluateTopicModelResultTaskRequest(
    project_id=cache.config.project_id,
    column=column.name,
  ))

@router.post('/experiment/start')
def post__topic_experiment(
  cache: ProjectCacheDependency,
  column: TextualSchemaColumnDependency,
  topic_modeling_result: TopicModelingResultDependency,
  body: TopicModelExperimentSchema
):
  return perform_topic_model_experiment(BERTopicExperimentTaskRequest(
    project_id=cache.config.project_id,
    column=column.name,
    constraint=body.constraint,
    n_trials=body.n_trials,
  ))

@router.get('/experiment/status')
def get__topic_experiment_status(
  cache: ProjectCacheDependency,
  column: TextualSchemaColumnDependency,
  topic_modeling_result: TopicModelingResultDependency,
):
  return check_topic_model_experiment_status(BERTopicExperimentTaskRequest(
    project_id=cache.config.project_id,
    column=column.name,
    constraint=None, # type: ignore
    n_trials=0, # type: ignore
  ))