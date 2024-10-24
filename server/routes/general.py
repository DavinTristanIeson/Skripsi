from fastapi import APIRouter

from common.models.api import ApiError, ApiResult
from common.models.enum import ExposedEnum


router = APIRouter()

@router.get('/enums')
def get__enums():
  return ApiResult(data=ExposedEnum().get_all_enums(), message=None)

@router.get('/enums/{enum_name}')
def get__enum(enum_name: str):
  enum = ExposedEnum().get_enum(enum_name)
  if enum is None:
    raise ApiError(f"Enum {enum_name} does not exist.", 404)
  enum_fields = list(map(lambda kv: dict(label=kv[1].label, value=kv[0], description=kv[1].description), enum.labels.items()))
  return ApiResult(data=enum_fields, message=None)