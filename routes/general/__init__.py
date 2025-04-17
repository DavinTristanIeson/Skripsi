from fastapi import APIRouter

from modules.api import ApiResult, ExposedEnum

router = APIRouter(
  tags=["General"]
)

@router.get('/enums')
def get__enums():
  return ApiResult(data=ExposedEnum().get_all_enums(), message=None)