from typing import Any
from fastapi import APIRouter

from controllers.table.comparison import compare_group_words
from controllers.table.filter import get_column_topic_words, get_column_word_frequencies
from models.table import ComparisonGroupWordsSchema, GetTableColumnSchema, GetTableGeographicalColumnSchema, ComparisonStatisticTestSchema, TableColumnCountsResource, TableColumnFrequencyDistributionResource, TableColumnGeographicalPointsResource, TableColumnValuesResource, TableTopicsResource, TableWordCloudResource
from modules.api.wrapper import ApiResult
from modules.comparison.engine import TableComparisonResult
from modules.table import PaginationParams

from controllers.project import ProjectCacheDependency

from controllers.table import (
  paginate_table, get_column_counts,
  get_column_unique_values, get_column_values,
  get_column_frequency_distribution, statistic_test,
  get_column_geographical_points,
)
from modules.table.pagination import TablePaginationApiResult

router = APIRouter(
  tags=['Table']
)

@router.post("/")
async def post__get_table(params: PaginationParams, cache: ProjectCacheDependency)->TablePaginationApiResult[dict[str, Any]]:
  return paginate_table(params, cache)

@router.post("/column/values")
async def post__get_table_column(body: GetTableColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableColumnValuesResource]:
  return get_column_values(body, cache)

@router.post("/column/frequency-distribution")
async def post__get_table_column__frequency_distribution(body: GetTableColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableColumnFrequencyDistributionResource]:
  return get_column_frequency_distribution(body, cache)

@router.post("/column/counts")
async def post__get_table_column__counts(body: GetTableColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableColumnCountsResource]:
  return get_column_counts(body, cache)

@router.post("/column/geographical")
async def post__get_table_column__geographical(body: GetTableGeographicalColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableColumnGeographicalPointsResource]:
  return get_column_geographical_points(body, cache)

@router.post("/column/unique")
async def post__get_table_column__unique(body: GetTableColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableColumnValuesResource]:
  return get_column_unique_values(body, cache)

@router.post("/column/word-frequencies")
async def post__get_table_column__word_frequencies(body: GetTableColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableWordCloudResource]:
  return get_column_word_frequencies(body, cache)

@router.post("/column/topic-words")
async def post__get_table_column__topic_words(body: GetTableColumnSchema, cache: ProjectCacheDependency)->ApiResult[TableTopicsResource]:
  return get_column_topic_words(body, cache)

@router.post("/statistic-test")
async def post__statistic_test(body: ComparisonStatisticTestSchema, cache: ProjectCacheDependency)->ApiResult[TableComparisonResult]:
  return statistic_test(body, cache)

@router.post("/compare-group-words")
async def post__compare_group_words(body: ComparisonGroupWordsSchema, cache: ProjectCacheDependency)->ApiResult[TableTopicsResource]:
  return compare_group_words(body, cache)
