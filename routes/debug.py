from fastapi import APIRouter

from common.task import TaskServer, TaskResponse
from common.models.api import ApiResult


router = APIRouter(
  tags=["Debug"]
)

@router.post('/sanity-check')
def post__sanity_check(id: str):
  server = TaskServer()
  sanity = TaskResponse.Idle(id)
  with server.lock:
    server.results[id] = sanity
  return ApiResult(data=server.result(id), message=None)


@router.get('/task-state')
def get__task_state():
  return ApiResult(data=TaskServer().results, message=None)

@router.get('/result/{id}')
def get__result(id: str):
  return ApiResult(data=TaskServer().result(id), message=None)