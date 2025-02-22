from fastapi import APIRouter

from models.table import GetTableColumnSchema, GetTableGeographicalColumnSchema, TableColumnsStatisticTestSchema
from modules.table import PaginationParams

from controllers.project import ProjectCacheDependency

from controllers.table import (
  paginate_table, get_column_counts,
  get_column_unique_values, get_column_values,
  get_column_frequency_distribution, statistic_test,
  get_column_geographical_points,
)

router = APIRouter(
  tags=['Table']
)

@router.post("/")
async def post__get_table(params: PaginationParams, cache: ProjectCacheDependency):
  return paginate_table(params, cache)

@router.post("/column/values")
async def post__get_table_column(body: GetTableColumnSchema, cache: ProjectCacheDependency):
  return get_column_values(body, cache)

@router.post("/column/frequency-distribution")
async def post__get_table_column__frequency_distribution(body: GetTableColumnSchema, cache: ProjectCacheDependency):
  return get_column_frequency_distribution(body, cache)

@router.post("/column/counts")
async def post__get_table_column__counts(body: GetTableColumnSchema, cache: ProjectCacheDependency):
  return get_column_counts(body, cache)

@router.post("/column/geographical")
async def post__get_table_column__geographical(body: GetTableGeographicalColumnSchema, cache: ProjectCacheDependency):
  return get_column_geographical_points(body, cache)

@router.post("/column/unique")
async def post__get_table_column__unique(body: GetTableColumnSchema, cache: ProjectCacheDependency):
  return get_column_unique_values(body, cache)

@router.post("/statistic-test")
async def post__statistic_test(body: TableColumnsStatisticTestSchema, cache: ProjectCacheDependency):
  return statistic_test(body, cache)
