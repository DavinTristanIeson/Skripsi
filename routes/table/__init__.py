from typing import Any
from fastapi import APIRouter

from routes.table.controller.comparison import compare_group_words
from routes.table.controller.filter import get_column_geographical_aggregate_values, get_column_paired_values
from routes.table.model import (
  CompareSubdatasetsSchema, DatasetFilterSchema, GetTableColumnAggregateValuesSchema, GetTableColumnSchema, GetTableGeographicalAggregateValuesSchema,
  GetTableGeographicalColumnSchema, GetTablePairedColumnSchema, TableColumnAggregateValuesResource,
  TableColumnCountsResource, TableColumnFrequencyDistributionResource,
  TableColumnGeographicalPointsResource, TableColumnPairedValuesResource, TableColumnValuesResource, TableDescriptiveStatisticsResource, TableTopicsResource,
  TableWordFrequenciesResource
)
from modules.api.wrapper import ApiResult
from modules.table import PaginationParams

from routes.dependencies.project import ProjectCacheDependency

from .controller import (
  paginate_table, get_column_counts,
  get_column_frequency_distribution,
  get_column_geographical_points, get_column_aggregate_values,
  get_column_descriptive_statistics, get_column_topic_words,
  get_column_unique_values, get_column_values, get_column_word_frequencies,
  get_affected_rows
)
from modules.table.filter_variants import TableFilter
from modules.table.pagination import TablePaginationApiResult

router = APIRouter(
  tags=['Table']
)

@router.post("/check-filter")
async def post__check_filter(filter: TableFilter, cache: ProjectCacheDependency)->ApiResult[TableFilter]:
  return ApiResult(data=filter, message=None)

@router.post("/")
async def post__get_table(params: PaginationParams, cache: ProjectCacheDependency)->TablePaginationApiResult[dict[str, Any]]:
  response = paginate_table(params, cache)
  return response

@router.post("/affected-rows")
async def post__get_affected_rows(params: DatasetFilterSchema, cache: ProjectCacheDependency)->ApiResult[list[int]]:
  return get_affected_rows(params, cache)

@router.post("/column/frequency-distribution")
async def post__get_table_column__frequency_distribution(body: GetTableColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableColumnFrequencyDistributionResource]:
  return get_column_frequency_distribution(body, cache)

@router.post("/column/aggregate-values")
async def post__get_table_column__aggregate_values(body: GetTableColumnAggregateValuesSchema, cache: ProjectCacheDependency)->ApiResult[TableColumnAggregateValuesResource]:
  return get_column_aggregate_values(body, cache)

@router.post("/column/counts")
async def post__get_table_column__counts(body: GetTableColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableColumnCountsResource]:
  return get_column_counts(body, cache)

@router.post("/column/values")
async def post__get_table_column(body: GetTableColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableColumnValuesResource]:
  return get_column_values(body, cache)

@router.post("/column/paired-values")
async def post__get_table_paired_column(body: GetTablePairedColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableColumnPairedValuesResource]:
  return get_column_paired_values(body, cache)

@router.post("/column/geographical")
async def post__get_table_column__geographical(body: GetTableGeographicalColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableColumnGeographicalPointsResource]:
  return get_column_geographical_points(body, cache)

@router.post("/column/geographical/aggregate-values")
async def post__get_table_column__geographical__aggregate_values(body: GetTableGeographicalAggregateValuesSchema, cache: ProjectCacheDependency)->ApiResult[TableColumnGeographicalPointsResource]:
  return get_column_geographical_aggregate_values(body, cache)

@router.post("/column/descriptive-statistics")
async def post__get_table_column__descriptive_statistics(body: GetTableColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableDescriptiveStatisticsResource]:
  return get_column_descriptive_statistics(body, cache)

@router.post("/column/unique")
async def post__get_table_column__unique(body: GetTableColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableColumnValuesResource]:
  return get_column_unique_values(body, cache)

@router.post("/column/word-frequencies")
async def post__get_table_column__word_frequencies(body: GetTableColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableWordFrequenciesResource]:
  return get_column_word_frequencies(body, cache)

@router.post("/column/topic-words")
async def post__get_table_column__topic_words(body: GetTableColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableTopicsResource]:
  return get_column_topic_words(body, cache)

@router.post("/compare/words")
async def post__compare_group_words(body: CompareSubdatasetsSchema, cache: ProjectCacheDependency)->ApiResult[TableTopicsResource]:
  return compare_group_words(body, cache)