from models.table import DatasetFilterSchema

from modules.api.wrapper import ApiResult
from modules.project.cache import ProjectCache
from modules.table.engine import TableEngine
from modules.table.pagination import PaginationParams, TablePaginationApiResult


def paginate_table(params: PaginationParams, cache: ProjectCache)->TablePaginationApiResult:
  engine = TableEngine(cache.config)
  data, meta = engine.paginate_workspace(params)
  return TablePaginationApiResult(
    data=data.to_dict(orient="records"),
    message=None,
    columns=cache.config.data_schema.columns,
    meta=meta
  )

def get_affected_rows(params: DatasetFilterSchema, cache: ProjectCache)->ApiResult[list[int]]:
  df = cache.load_workspace()
  engine = TableEngine(cache.config)
  data = engine.filter(df, params.filter)
  return ApiResult(data=list(map(int, data.index)), message=None)

__all__ = [
  "paginate_table",
  "get_affected_rows",
]