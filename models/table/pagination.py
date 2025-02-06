from typing import Generic, Optional, TypeVar
import pydantic
from .filter import TableSort
from .filter_variants import TableFilter
from common.models.validators import CommonModelConfig

class PaginationParams(pydantic.BaseModel):
  page: int
  limit: int
  filter: list[TableFilter]
  sort: list[TableSort]

class PaginationMeta(PaginationParams, pydantic.BaseModel):
  pages: int
  total: int

T = TypeVar("T")
class PaginatedApiModel(Generic[T], pydantic.BaseModel):
  model_config = CommonModelConfig
  data: list[T]
  message: Optional[str]
  meta: PaginationMeta
