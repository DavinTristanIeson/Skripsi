from typing import Annotated, cast
from fastapi import APIRouter, Depends, Query

from common.ipc.operations import IPCOperationResponseData
from common.ipc.requests import IPCRequestData
from common.ipc.taskqueue import IPCTaskClient
from common.models.api import ApiError, ApiResult
from server.controllers.project_checks import PerformedTopicModelingDependency, ProjectExistsDependency, SchemaColumnExistsDependency, get_data_column
from wordsmith.data.cache import ProjectCacheManager
from wordsmith.data.config import Config
from wordsmith.data.schema import SchemaColumnTypeEnum, SchemaColumn


router = APIRouter(
  tags=["Association"]
)

def check_association_columns(config: ProjectExistsDependency, column1: str = Query(), column2: str = Query()):
  col1 = get_data_column(config, column1)
  col2 = get_data_column(config, column2)
  if (col1.type != SchemaColumnTypeEnum.Textual):
    raise ApiError("The first column should be a column of type 'Textual'.", 400)
  if (col2.type == SchemaColumnTypeEnum.Unique):
    raise ApiError("The second column should not be a column of type 'Unique'.", 400)
  
  workspace = ProjectCacheManager().workspaces.get(config.project_id)
  if col1.name not in workspace.columns:
    raise ApiError(f"Column {col1.name} could not be found in the current workspace. Please run the topic modeling procedure again to make sure that the workspace table is updated to the new dataset.", 404)
  if col2.name not in workspace.columns:
    raise ApiError(f"Column {col2.name} could not be found in the current workspace. Please run the topic modeling procedure again to make sure that the workspace table is updated to the new dataset.", 404)
  
  return col1, col2


AssociationColumnsExistsDependency = Annotated[tuple[SchemaColumn, SchemaColumn], Depends(check_association_columns)]

@router.get('/{project_id}/association')
async def get__association(config: ProjectExistsDependency, columns: AssociationColumnsExistsDependency):
  column1, column2 = columns
  
  task_id = IPCRequestData.Association.task_id(config.project_id, column1.name, column2.name)
  locker = IPCTaskClient()

  if result:=locker.result(task_id):
    return result

  raise ApiError(f"No topic association task has been started for {config.project_id}.", 400)

@router.post('/{project_id}/association/start')
async def post__request_association(task: PerformedTopicModelingDependency, config: ProjectExistsDependency, columns: AssociationColumnsExistsDependency):
  column1, column2 = columns
  task_id = IPCRequestData.Association.task_id(config.project_id, column1.name, column2.name)
  locker = IPCTaskClient()

  if result:=locker.result(task_id):
    return result

  result = locker.request(IPCRequestData.Association(
    id=task_id,
    project_id=config.project_id,
    column1=column1.name,
    column2=column2.name,
  ))
  result = cast(IPCOperationResponseData.Result, result)

  return ApiResult(data=None, message=f"Please wait for a few seconds while we create the visualizations for the relationship between {column1.name} and {column2.name} in {config.project_id}")