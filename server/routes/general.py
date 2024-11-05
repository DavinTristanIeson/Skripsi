from fastapi import APIRouter

from common.models.api import ApiResult
from common.models.enum import ExposedEnum


router = APIRouter()

@router.get('/enums')
def get__enums():
  return ApiResult(data=ExposedEnum().get_all_enums(), message=None)