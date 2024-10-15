from fastapi import APIRouter

import controllers.ipc as ipc
from models.server.api import ApiResult


router = APIRouter()

@router.post('/{id}')
def post__topic_modeling_request(id: str):
  ipc.IPCClient().send(ipc.IPCMessageVariant.TopicModelingRequest(
    id=id,
  ))
  return ApiResult(data=None, message=f"Started topic modeling task for {id}")