from typing import Generic, Optional, TypeVar
import pydantic

from modules.config.schema.schema_variants import SchemaColumn
from .filter import TableSort
from .filter_variants import TableFilter

class PaginationParams(pydantic.BaseModel):
  page: Optional[int] = pydantic.Field(default=0)
  limit: Optional[int] = pydantic.Field(default=15)
  filter: Optional[TableFilter] = None
  sort: Optional[TableSort] = None

class PaginationMeta(pydantic.BaseModel):
  pages: int
  total: int

T = TypeVar("T")
class TablePaginationApiResult(pydantic.BaseModel, Generic[T]):
  data: list[T]
  message: Optional[str]
  columns: list[SchemaColumn]
  meta: PaginationMeta

__all__ = [
  "TablePaginationApiResult",
  "PaginationMeta",
  "PaginationParams"
]