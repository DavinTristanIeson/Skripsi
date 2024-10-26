from fastapi import APIRouter

from common.ipc.operations import IPCOperationRequestData
from common.ipc.taskqueue import IPCTaskClient
from common.models.api import ApiResult


router = APIRouter(
  tags=["Debug"]
)

@router.get('/sanity-check')
def post__sanity_check(id: str):
  response = IPCTaskClient().operation(IPCOperationRequestData.SanityCheck(id=id))
  return ApiResult(data=response, message=None)


@router.get('/task-state')
def get__task_state():
  response = IPCTaskClient().operation(IPCOperationRequestData.TaskState())
  return ApiResult(data=response, message=None)

@router.get('/result/{id}')
def get__result(id: str):
  response = IPCTaskClient().result(id)
  return ApiResult(data=response, message=None)