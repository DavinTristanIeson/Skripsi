import functools
import http

from controllers.project import ProjectCacheDependency
from modules.exceptions.files import FileLoadingException
from routes.topic.model import StartTopicModelingSchema, TopicModelingTaskRequest
from modules.api.wrapper import ApiError, ApiResult
from modules.config import TextualSchemaColumn
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache, ProjectCacheManager
from modules.project.paths import ProjectPaths
from modules.task import (
  scheduler,
  TaskResponse
)
from modules.task.storage import AlternativeTaskResponse, TaskConflictResolutionBehavior, TaskStorage
from modules.topic.model import TopicModelingResult
from modules.topic.procedure import BERTopicProcedureFacade


logger = ProvisionedLogger().provision("Topic Controller")

def topic_modeling_task(payload: TopicModelingTaskRequest):
  taskstore = TaskStorage()
  with taskstore.proxy_context(payload.task_id) as proxy:
    facade = BERTopicProcedureFacade(
      task=proxy,
      column=payload.column,
      project_id=payload.project_id
    )
    facade.run()

def start_topic_modeling(options: StartTopicModelingSchema, cache: ProjectCacheDependency, column: TextualSchemaColumn):
  config = cache.config
  df = cache.workspaces.load()

  ProjectCacheManager().invalidate(config.project_id)

  cleanup_files: list[str] = []

  if not options.use_cached_umap_vectors:
    logger.info(f"Cleaning up cached UMAP embeddings from {column.name}.")
    cleanup_files.extend([
      ProjectPaths.VisualizationEmbeddings(column.name),
      ProjectPaths.UMAPEmbeddings(column.name),
    ])

  if not options.use_cached_document_vectors:
    cleanup_files.append(ProjectPaths.DocumentEmbeddings(column.name))
    logger.info(f"Cleaning up cached document embeddings from {column.name}.")
  

  if not options.use_preprocessed_documents:
    logger.info(f"Cleaning up cached preprocessed documents and topic column from {column.name}.")
    if column.preprocess_column.name in df.columns:
      df.drop(column.preprocess_column.name, axis=1, inplace=True)
    if column.topic_column.name in df.columns:
      df.drop(column.topic_column.name, axis=1, inplace=True)
    config.save_workspace(df)
  
  if not options.use_cached_document_vectors or not options.use_cached_umap_vectors or not options.use_preprocessed_documents:
    logger.info(f"Cleaning up BERTopic experiments from {column.name}.")
    cleanup_files.append(ProjectPaths.TopicModelExperiments(column.name))

  cleanup_files.append(ProjectPaths.Topics(column.name))
  ProjectCacheManager().invalidate(config.project_id)
  config.paths._cleanup(
    directories=[ProjectPaths.BERTopic(column.name)],
    files=cleanup_files,
  )

  request = TopicModelingTaskRequest(
    project_id=config.project_id,
    column=column.name,
  )

  store = TaskStorage()
  store.add_task(
    scheduler=scheduler,
    task_id=request.task_id,
    task=topic_modeling_task,
    args=[request],
    conflict_resolution=TaskConflictResolutionBehavior.Ignore,
    idle_message=f"Requested topic modeling algorithm to be applied to \"{column.name}\".",
  )
  
  return ApiResult(data=None, message=f"The topic modeling algorithm will soon be applied to \"{column.name}\". Please wait for a few seconds (or minutes depending on the size of your dataset) for the algorithm to complete.")

def __topic_modeling_status_alternative(cache: ProjectCache, column: TextualSchemaColumn):
  try:
    topics = cache.topics.load(column.name)
  except FileLoadingException:
    topics = None
  if topics is None:
    return None
  return AlternativeTaskResponse(
    data=topics,
    message="The topic modeling procedure has already been executed before. Feel free to explore the discovered topics."
  )

def check_topic_modeling_status(cache: ProjectCacheDependency, column: TextualSchemaColumn)->TaskResponse[TopicModelingResult]:
  config = cache.config

  request = TopicModelingTaskRequest(
    project_id=config.project_id,
    column=column.name,
  )

  store = TaskStorage()

  alternative_response = functools.partial(__topic_modeling_status_alternative, cache, column)
  
  task_result = store.get_task_result(
    task_id=request.task_id,
    alternative_response=alternative_response,
  )
  if task_result is None:
    raise ApiError(f"No topic modeling task has been started for \"{column.name}\" in project \"{config.metadata.name}\".", http.HTTPStatus.BAD_REQUEST)
  return task_result


__all__ = [
  "start_topic_modeling",
  "check_topic_modeling_status"
]