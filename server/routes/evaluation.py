from fastapi import APIRouter

from common.ipc.requests import IPCRequest, IPCRequestData
from common.ipc.taskqueue import IPCTaskClient
from common.models.api import ApiError, ApiResult
from server.controllers.project_checks import ProjectExistsDependency, SchemaColumnExistsDependency
from wordsmith.data.schema import SchemaColumnTypeEnum


router = APIRouter(
  tags=["Table"]
)

@router.post("/{project_id}/evaluation/start")
def post__start_topic_evaluation(config: ProjectExistsDependency, column: SchemaColumnExistsDependency):
  client = IPCTaskClient()
  task_id = IPCRequestData.Evaluation.task_id(config.project_id, column.name)
  
  client.request(IPCRequestData.Evaluation(
    id=task_id,
    project_id=config.project_id,
    column=column.name,
  ))

  return ApiResult(
    data=None,
    message=f"The topic evaluation procedure has been started for {column.name}. We need to scan the words in all of the documents, so this may take a few minutes depending on the size of your dataset."
  )

@router.get("/{project_id}/evaluation")
def get__topic_evaluation(config: ProjectExistsDependency, column: SchemaColumnExistsDependency):
  if column.type != SchemaColumnTypeEnum.Textual:
    raise ApiError("Only textual columns can have their topics evaluated.", 400)
  
  client = IPCTaskClient()
  task_id = IPCRequestData.Evaluation.task_id(config.project_id, column.name)
  if result := client.result(task_id):
    return result

  result = config.paths.load_evaluation(column.name)
  return ApiResult(
    data=result,
    message=""
  )