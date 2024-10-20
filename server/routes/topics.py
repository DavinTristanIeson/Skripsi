from fastapi import APIRouter

from common.ipc.requests import IPCRequestData
from common.ipc.taskqueue import IPCTaskLocker
from common.models.api import ApiResult


router = APIRouter(
  tags=["Topics"]
)

@router.post('/{project_id}')
def post__topic_modeling_request(project_id: str):
  locker = IPCTaskLocker()

  task_id = f"topic-modeling-{project_id}" # Unique but deterministic key
  if result:=locker.result(task_id):
    return ApiResult(data=result, message=result.message)

  IPCTaskLocker().request(IPCRequestData.TopicModeling(
    id=task_id,
    project_id=project_id
  ))
  return ApiResult(data=None, message=f"Started topic modeling task for {project_id}")