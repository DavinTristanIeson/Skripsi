from fastapi import APIRouter

from common.ipc.requests import IPCRequestData
from common.ipc.tasks import IPCTaskLocker
from common.models.api import ApiResult


router = APIRouter()

@router.post('/{id}')
def post__topic_modeling_request(id: str):
  IPCTaskLocker().request(IPCRequestData.TopicModeling(
    project_id=id
  ))
  return ApiResult(data=None, message=f"Started topic modeling task for {id}")