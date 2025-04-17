import http

from apscheduler.jobstores.base import JobLookupError

from controllers.project import ProjectCacheDependency
from routes.topic.model import StartTopicModelingSchema, TopicModelingTaskRequest
from modules.api.wrapper import ApiError, ApiResult
from modules.config import TextualSchemaColumn
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCacheManager
from modules.project.paths import ProjectPaths
from modules.task import (
  scheduler,
  TaskLog, TaskResponse, 
  TaskStatusEnum
)
from modules.task.storage import TaskStorage
from modules.topic.model import TopicModelingResult
from modules.topic.procedure import BERTopicProcedureFacade


logger = ProvisionedLogger().provision("Topic Controller")

def topic_modeling_task(payload: TopicModelingTaskRequest):
  proxy = TaskStorage().proxy(payload.task_id)
  with proxy.lock:
    proxy.initialize()
    proxy.task.status = TaskStatusEnum.Pending
  facade = BERTopicProcedureFacade(
    task=proxy,
    column=payload.column,
    project_id=payload.project_id
  )
  try:
    facade.run()
  except Exception as e:
    logger.exception(e)
    proxy.log_error(f"We weren't able to complete the topic modeling procedure due to the following error: {e}")
    with proxy.lock:
      proxy.task.status = TaskStatusEnum.Failed

def start_topic_modeling(options: StartTopicModelingSchema, cache: ProjectCacheDependency, column: TextualSchemaColumn):
  config = cache.config
  df = cache.load_workspace()

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

  has_pending_task = False
  try:
    scheduler.remove_job(request.task_id)
    has_pending_task = True
  except JobLookupError:
    pass

  project_id = config.project_id
  scheduler.add_job(topic_modeling_task, args=[request], max_instances=1)

  if has_pending_task:
    return ApiResult(data=None, message=f"The topic modeling algorithm will soon be applied to Project \"{project_id}\". Please wait for a few seconds (or minutes depending on the size of your dataset) for the algorithm to complete.")
  else:
    return ApiResult(data=None, message=f"The topic modeling algorithm will be applied again to Project \"{project_id}\"; meanwhile, the previous pending topic modeling task will be canceled. Please wait for a few seconds (or minutes depending on the size of your dataset) for the algorithm to complete.")
  

def check_topic_modeling_status(cache: ProjectCacheDependency, column: TextualSchemaColumn)->TaskResponse[TopicModelingResult]:
  config = cache.config

  request = TopicModelingTaskRequest(
    project_id=config.project_id,
    column=column.name,
  )

  store = TaskStorage()
  with store.lock:
    result = store.results.get(request.task_id, None)
  
  if result is not None:
    return result

  try:
    topics = cache.load_topic(column.name)
  except ApiError:
    topics = None
  
  if topics is not None:
    response = TaskResponse(
      id=request.task_id,
      logs=[
        TaskLog(
          status=TaskStatusEnum.Success,
          message="The topic modeling procedure has already been executed before. Feel free to explore the discovered topics.",
        )
      ],
      data=topics,
      status=TaskStatusEnum.Success
    )
    with store.lock:
      store.results[request.task_id] = response
    return response
  
  raise ApiError(f"No topic modeling task has been started for \"{column.name}\" in project \"{config.project_id}\".", http.HTTPStatus.BAD_REQUEST)

__all__ = [
  "start_topic_modeling",
  "check_topic_modeling_status"
]