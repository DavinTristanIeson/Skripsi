from typing import Annotated, Optional

from fastapi import Depends, Query
import pandas as pd
from common.ipc.requests import IPCRequestData
from common.ipc.responses import IPCResponse
from common.ipc.task import IPCTask
from common.ipc.taskqueue import IPCTaskClient
from common.models.api import ApiError
import os

from server.models.project import ProjectTaskStatus
from wordsmith.data.cache import ProjectCacheManager
from wordsmith.data.config import Config
from wordsmith.data.paths import ProjectPaths
from wordsmith.data.schema import SchemaColumn

def get_project_config(project_id: str):
  config = ProjectCacheManager().configs.get(project_id)
  config.project_id = project_id
  return config
ProjectExistsDependency = Annotated[Config, Depends(get_project_config)]

def get_data_column(config: ProjectExistsDependency, column: str = Query()):
  try:
    return config.data_schema.assert_exists(column)
  except KeyError:
    raise ApiError(f"Column {column} doesn't exist in the schema. Please make sure that your schema is properly configured to your data.", 404)
SchemaColumnExistsDependency = Annotated[SchemaColumn, Depends(get_data_column)]

def check_topic_modeling_status(config: ProjectExistsDependency, project_id: str):
  locker = IPCTaskClient()
  task_id = IPCRequestData.TopicModeling.task_id(project_id)
  task = locker.result(task_id)

  if task is not None:
    if task.status == ProjectTaskStatus.Idle or task.status == ProjectTaskStatus.Pending:
      raise ApiError("We are currently performing the topic modeling procedure on your dataset. Please check again later!", 400)
    if task.status == ProjectTaskStatus.Failed:
      raise ApiError("Oh no, it seems that the topic modeling procedure has failed with an unexpected error.", 400)

  if not os.path.exists(config.paths.full_path(ProjectPaths.BERTopic)):
    raise ApiError("The topic modeling procedure has not been performed. If our interface did not automatically start the topic modeling procedure, please manually run it again.", 400)
  
  if task is None:
    return None
  return task.status
PerformedTopicModelingDependency = Annotated[Optional[IPCResponse], Depends(check_topic_modeling_status)]

def get_workspace_table(config: ProjectExistsDependency, project_id: str)->pd.DataFrame:
  locker = IPCTaskClient()
  task_id = IPCRequestData.TopicModeling.task_id(project_id)

  task = locker.result(task_id)
  
  try:
    workspace = config.paths.load_workspace()
    return workspace
  except ApiError:
    if task is not None:
      if task.status == ProjectTaskStatus.Idle:
        raise ApiError("Your dataset has not been preprocessed yet. Please wait while we finish other tasks.", 400)
      if task.status == ProjectTaskStatus.Pending:
        raise ApiError("We are currently preprocessing your dataset.", 400)
      
    raise ApiError("The topic modeling procedure has not been performed, and so the table is probably not preprocessed. If our interface did not automatically start the topic modeling procedure, please manually run it again.", 400)
PerformedTablePreprocessingDependency = Annotated[pd.DataFrame, Depends(get_workspace_table)]

__all__ = [
  "ProjectExistsDependency",
  "SchemaColumnExistsDependency",
  "PerformedTopicModelingDependency",
  "PerformedTablePreprocessingDependency"
]