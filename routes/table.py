from typing import Optional
from fastapi import APIRouter

from controllers.table.filter import get_column_counts
from models.table import GetTableColumnSchema, GetTableGeographicalColumnSchema
from modules.table import PaginationParams
from controllers.project.dependency import ProjectCacheDependency
from modules.table import TableEngine
from modules.table.pagination import PaginatedApiResult, PaginationMeta

from controllers.table import (
  SchemaColumnExistsDependency, paginate_table,
  get_column_values, get_column_frequency_distribution,
  get_column_geographical_points
)

router = APIRouter(
  tags=['Table']
)

@router.post("/")
async def post__get_table(params: PaginationParams, cache: ProjectCacheDependency):
  return paginate_table(params, cache)

@router.post("/column/values")
async def post__get_table_column(body: GetTableColumnSchema, cache: ProjectCacheDependency, column: SchemaColumnExistsDependency):
  return get_column_values(body, cache)

@router.post("/column/frequency-distribution")
async def post__get_table_column__frequency_distribution(body: GetTableColumnSchema, cache: ProjectCacheDependency, column: SchemaColumnExistsDependency):
  return get_column_frequency_distribution(body, cache)

@router.post("/column/counts")
async def post__get_table_column__counts(body: GetTableColumnSchema, cache: ProjectCacheDependency, column: SchemaColumnExistsDependency):
  return get_column_counts(body, cache)

@router.post("/column/geographical")
async def post__get_table_column__geographical(body: GetTableGeographicalColumnSchema, cache: ProjectCacheDependency, column: SchemaColumnExistsDependency):
  return get_column_geographical_points(body, cache)
