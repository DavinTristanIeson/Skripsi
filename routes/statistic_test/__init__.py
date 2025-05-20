from fastapi import APIRouter

from modules.api.wrapper import ApiResult

from modules.comparison import statistic_test
from modules.comparison.engine import StatisticTestResult
from routes.dependencies.project import ProjectCacheDependency
from .controller import binary_statistic_test_on_contingency_table, binary_statistic_test_on_distribution, contingency_table

from .model import (
  BinaryStatisticTestOnContingencyTableMainResource,
  BinaryStatisticTestOnDistributionResource,
  BinaryStatisticTestSchema,
  ContingencyTableResource,
  GetContingencyTableSchema,
  StatisticTestSchema,
)

router = APIRouter(
  tags=['Statistic Test']
)

@router.post("/test")
async def post__statistic_test(body: StatisticTestSchema, cache: ProjectCacheDependency)->ApiResult[StatisticTestResult]:
  return statistic_test(body, cache)

@router.post("/binary/test-distribution")
def post__test_distribution(body: BinaryStatisticTestSchema, cache: ProjectCacheDependency)->ApiResult[BinaryStatisticTestOnDistributionResource]:
  return ApiResult(data=binary_statistic_test_on_distribution(cache, body), message=None)

@router.post("/contingency-table")
def post__topics_contingency_table(body: GetContingencyTableSchema, cache: ProjectCacheDependency)->ApiResult[ContingencyTableResource]:
  return ApiResult(data=contingency_table(cache, body), message=None)
  
@router.post("/binary/test-contingency-table")
def post__test_contingency_table(body: GetContingencyTableSchema, cache: ProjectCacheDependency)->ApiResult[BinaryStatisticTestOnContingencyTableMainResource]:
  return ApiResult(data=binary_statistic_test_on_contingency_table(cache, body), message=None)