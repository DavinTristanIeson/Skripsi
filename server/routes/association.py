import os
from typing import cast
from fastapi import APIRouter

from common.ipc.operations import IPCOperationResponseData
from common.ipc.requests import IPCRequestData
from common.ipc.taskqueue import IPCTaskClient
from common.models.api import ApiError, ApiResult
from server.controllers.project_checks import ProjectExistsDependency, SchemaColumnExistsDependency
from wordsmith.data.config import Config
from wordsmith.data.schema import SchemaColumnTypeEnum


router = APIRouter(
  tags=["Association"]
)
@router.get('/{project_id}/association')
async def get__association(config: ProjectExistsDependency, column1: SchemaColumnExistsDependency, column2: SchemaColumnExistsDependency):
  if (column1.type is not SchemaColumnTypeEnum.Textual):
    raise ApiError("Please fill column1 with textual type of column", 400)
  if (column2.type is SchemaColumnTypeEnum.Unique):
    raise ApiError("Please fill column2 with non-unique type of column", 400)

  workspace = config.paths.load_workspace()
  if column1.name not in workspace.columns:
    raise ApiError(f"Column {column1.name} could not be found in the current workspace. Please run the topic modeling procedure again to make sure that the workspace table is updated to the new dataset.", 404)
  if column2.name not in workspace.columns:
    raise ApiError(f"Column {column2.name} could not be found in the current workspace. Please run the topic modeling procedure again to make sure that the workspace table is updated to the new dataset.", 404)
  
  task_id = IPCRequestData.AssociationPlot.task_id(config.project_id, column1.name, column2.name)
  locker = IPCTaskClient()

  if result:=locker.result(task_id):
    return result

  result = locker.request(IPCRequestData.AssociationPlot(
    id=task_id,
    project_id=config.project_id,
    column1=column1.name,
    column2=column2.name,
  ))
  result = cast(IPCOperationResponseData.Result, result)

  return result.data