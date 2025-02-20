from typing import Generic, Optional, TypeVar
import pydantic
from .filter import TableSort
from .filter_variants import TableFilter

class PaginationParams(pydantic.BaseModel):
  page: Optional[int] = pydantic.Field(default=0)
  limit: Optional[int] = pydantic.Field(default=15)
  filter: Optional[TableFilter] = None
  sort: Optional[TableSort] = None

class PaginationMeta(PaginationParams, pydantic.BaseModel):
  pages: int
  total: int

T = TypeVar("T")
class PaginatedApiResult(pydantic.BaseModel, Generic[T], ):
  data: list[T]
  message: Optional[str]
  meta: PaginationMeta

__all__ = [
  "PaginatedApiResult",
  "PaginationMeta",
  "PaginationParams"
]