from fastapi import APIRouter

from common.ipc.requests import IPCRequest, IPCRequestData
from common.ipc.responses import IPCResponse, IPCResponseData, IPCResponseStatus
from common.ipc.taskqueue import IPCTaskClient
from common.models.api import ApiError, ApiResult
from server.controllers.project_checks import ProjectExistsDependency, SchemaColumnExistsDependency
from wordsmith.data.schema import SchemaColumnTypeEnum


router = APIRouter(
  tags=["Table"]
)

@router.post("/{project_id}/evaluation/start")
def post__start_topic_evaluation(config: ProjectExistsDependency, column: SchemaColumnExistsDependency):
  if column.type != SchemaColumnTypeEnum.Textual:
    raise ApiError("Only textual columns can have their topics evaluated.", 400)
  
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
  if column.name in result.root:
    return IPCResponse(
      data=IPCResponseData.Evaluation.model_validate({
        **result.root[column.name].model_dump()
      }),
      id=task_id,
      message=f"The topics of {column.name} has been successfully evaluated. Check out the quality of the topics discovered by the topic modeling algorithm with these scores; even though they may be harder to interpret than classification scores like accuracy or precision.",
      progress=1,
      status=IPCResponseStatus.Success,
    )
  
  raise ApiError(f"The topics in {column.name} has yet to be evaluated. Please run the topic modeling procedure first before continuing.", 400)