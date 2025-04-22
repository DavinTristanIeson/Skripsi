from fastapi import APIRouter

from modules.api.wrapper import ApiResult

from controllers.project import ProjectCacheDependency
from routes.correlation.controller import binary_statistic_test_on_contingency_table, binary_statistic_test_on_distribution, contingency_table

from .model import (
  BinaryStatisticTestOnDistributionResource,
  BinaryStatisticTestSchema,
  ContingencyTableResource,
  BinaryStatisticTestOnContingencyTableResource,
  TopicCorrelationSchema
)

router = APIRouter(
  tags=['Correlation']
)

@router.post("/binary/test-distribution")
def post__test_distribution(body: BinaryStatisticTestSchema, cache: ProjectCacheDependency)->ApiResult[list[BinaryStatisticTestOnDistributionResource]]:
  return ApiResult(data=binary_statistic_test_on_distribution(cache, body), message=None)

@router.post("/contingency-table")
def post__topics_contingency_table(body: TopicCorrelationSchema, cache: ProjectCacheDependency)->ApiResult[ContingencyTableResource]:
  return ApiResult(data=contingency_table(cache, body), message=None)
  
@router.post("/binary/test-contingency-table")
def post__test_contingency_table(body: BinaryStatisticTestSchema, cache: ProjectCacheDependency)->ApiResult[list[BinaryStatisticTestOnContingencyTableResource]]:
  return ApiResult(data=binary_statistic_test_on_contingency_table(cache, body), message=None)