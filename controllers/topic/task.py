import http
import itertools
import os
from typing import cast
from controllers.project.dependency import ProjectCacheDependency
from models.topic import StartTopicModelingSchema
from modules.api.wrapper import ApiError, ApiResult
from modules.config import ProjectPaths, SchemaColumnTypeEnum, TextualSchemaColumn
from modules.logger.provisioner import ProvisionedLogger
from modules.task import (
  TaskEngine, TaskRequest, TaskRequestData,
  TaskLog, TaskResponse, TaskResponseData,
  TaskStatusEnum
)
from modules.task.requests import TaskRequestType
from modules.topic.procedure.procedure import bertopic_topic_modeling


logger = ProvisionedLogger().provision("Topic Controller")
def start_topic_modeling(options: StartTopicModelingSchema, cache: ProjectCacheDependency):
  config = cache.config

  if options.targets is None:
    textual_columns = config.data_schema.textual()
    if len(textual_columns) == 0:
      raise ApiError("There are no textual columns to perform topic modeling on. Please modify your configuration and specify at least one textual column so that we may perform topic modeling on it.", http.HTTPStatus.BAD_REQUEST)
  else:
    textual_columns = cast(list[TextualSchemaColumn], list(map(
      lambda col: config.data_schema.assert_of_type(col, [SchemaColumnTypeEnum.Textual]),
      options.targets
    )))
    if len(textual_columns) == 0:
      raise ApiError("Please specify at least one textual column to perform topic modeling on. Alternatively, select all columns to run the topic modeling algorithm on all columns at once.", http.HTTPStatus.BAD_REQUEST) 

  targets = list(map(lambda col: col.name, textual_columns))

  if not options.use_cached_umap_embeddings:
    logger.info(f"Cleaning up cached UMAP embeddings from {targets}.")
    config.paths._cleanup(
      directories=list(itertools.chain(
        map(
          lambda column: ProjectPaths.VisualizationEmbeddings(column.name),
          textual_columns
        ),
        map(
          lambda column: ProjectPaths.UMAPEmbeddings(column.name),
          textual_columns
        ),
      )),
      files=[]
    )

  if not options.use_cached_document_embeddings:
    logger.info(f"Cleaning up cached document embeddings from {targets}.")
    config.paths._cleanup(
      directories=list(map(
        lambda column: ProjectPaths.DocumentEmbeddings(column.name),
        textual_columns
      )),
      files=[]
    )

  df = cache.load_workspace()
  if not options.use_preprocessed_documents:
    logger.info(f"Cleaning up cached preprocessed document columns from {targets}.")
    for column in textual_columns:
      df.drop(column.preprocess_column.name, axis=1, inplace=True)

  logger.info(f"Cleaning up topic columns from {targets}.")
  for column in textual_columns:
    df.drop(column.topic_column.name, axis=1, inplace=True)

  cache.save_workspace(df)

  project_id = config.project_id
  task_id = TaskRequestData.TopicModeling.task_id(project_id)

  engine = TaskEngine()
  has_pending_task = engine.result(task_id)

  if not options.use_cached_document_embeddings:
    pass

  # Always cancel old task
  engine.begin_task(TaskRequest(
    id=task_id,
    project_id=project_id,
    data=TaskRequestData.TopicModeling()
  ))

  if has_pending_task:
    return ApiResult(data=None, message=f"The topic modeling algorithm will soon be applied to Project \"{project_id}\". Please wait for a few seconds (or minutes depending on the size of your dataset) for the algorithm to complete.")
  else:
    return ApiResult(data=None, message=f"The topic modeling algorithm will be applied again to Project \"{project_id}\"; meanwhile, the previous pending topic modeling task will be canceled. Please wait for a few seconds (or minutes depending on the size of your dataset) for the algorithm to complete.")
  

def check_topic_modeling_status(cache: ProjectCacheDependency, column: TextualSchemaColumn):
  config = cache.config
  task_id = TaskRequestData.TopicModeling.task_id(config.project_id)
  engine = TaskEngine()
  if result:=engine.result(task_id):
    return result
  
  bertopic_path = config.paths.full_path(ProjectPaths.BERTopic(column.name))
  if os.path.exists(bertopic_path):
    response = TaskResponse(
      id=task_id,
      logs=[
        TaskLog(
          status=TaskStatusEnum.Success,
          message="The topic modeling procedure has already been executed before. Feel free to explore the discovered topics.",
        )
      ],
      data=TaskResponseData.Empty(),
      status=TaskStatusEnum.Success
    )
    with engine.lock:
      engine.results[task_id] = response
  
  raise ApiError(f"No topic modeling task has been started for Project \"{config.project_id}\".", http.HTTPStatus.BAD_REQUEST)

TaskEngine().register(TaskRequestType.TopicModeling, bertopic_topic_modeling)

__all__ = [
  "start_topic_modeling",
  "check_topic_modeling_status"
]