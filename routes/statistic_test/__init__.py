from statistics import linear_regression
from fastapi import APIRouter

from modules.api.wrapper import ApiResult

from modules.comparison.engine import StatisticTestResult
from modules.regression.results.base import BaseRegressionInput
from modules.regression.results.linear import LinearRegressionInput, LinearRegressionResult
from modules.regression.results.logistic import LogisticRegressionInput, LogisticRegressionResult, MultinomialLogisticRegressionInput, MultinomialLogisticRegressionResult
from modules.regression.results.ordinal import OrdinalRegressionInput, OrdinalRegressionResult
from routes.dependencies.project import ProjectCacheDependency
from .controller import (
  statistic_test, binary_statistic_test_on_contingency_table,
  binary_statistic_test_on_distribution, contingency_table,
  omnibus_statistic_test, subdataset_cooccurrence, pairwise_statistic_test,
  linear_regression, logistic_regression,
  ordinal_regression, multinomial_logistic_regression
)

from .model import (
  BinaryStatisticTestOnContingencyTableResultMainResource,
  BinaryStatisticTestOnContingencyTableSchema,
  BinaryStatisticTestOnDistributionResultResource,
  BinaryStatisticTestOnDistributionSchema,
  ContingencyTableResource,
  GetContingencyTableSchema,
  GetSubdatasetCooccurrenceSchema,
  OmnibusStatisticTestSchema,
  PairwiseStatisticTestResultResource,
  PairwiseStatisticTestSchema,
  StatisticTestSchema,
  SubdatasetCooccurrenceResource,
)

router = APIRouter(
  tags=['Statistic Test']
)

@router.post('')
async def post__statistic_test(body: StatisticTestSchema, cache: ProjectCacheDependency)->ApiResult[StatisticTestResult]:
  return ApiResult(data=statistic_test(body, cache), message=None)

@router.post("/pairwise")
async def post__pairwise_statistic_test(body: PairwiseStatisticTestSchema, cache: ProjectCacheDependency)->ApiResult[PairwiseStatisticTestResultResource]:
  return ApiResult(data=pairwise_statistic_test(cache, body), message=None)

@router.post("/omnibus")
async def post__omnibus_statistic_test(body: OmnibusStatisticTestSchema, cache: ProjectCacheDependency)->ApiResult[StatisticTestResult]:
  return ApiResult(data=omnibus_statistic_test(cache, body), message=None)

@router.post("/contingency-table")
def post__contingency_table(body: GetContingencyTableSchema, cache: ProjectCacheDependency)->ApiResult[ContingencyTableResource]:
  return ApiResult(data=contingency_table(cache, body), message=None)
  
@router.post("/binary-test-distribution")
def post__test_distribution(body: BinaryStatisticTestOnDistributionSchema, cache: ProjectCacheDependency)->ApiResult[BinaryStatisticTestOnDistributionResultResource]:
  return ApiResult(data=binary_statistic_test_on_distribution(cache, body), message=None)

@router.post("/binary-test-contingency-table")
def post__test_contingency_table(body: BinaryStatisticTestOnContingencyTableSchema, cache: ProjectCacheDependency)->ApiResult[BinaryStatisticTestOnContingencyTableResultMainResource]:
  return ApiResult(data=binary_statistic_test_on_contingency_table(cache, body), message=None)

@router.post("/co-occurrence")
async def post__cooccurrence(body: GetSubdatasetCooccurrenceSchema, cache: ProjectCacheDependency)->ApiResult[SubdatasetCooccurrenceResource]:
  return ApiResult(
    data=subdataset_cooccurrence(body, cache),
    message=None,
  )


@router.post("/regression/linear")
async def post__regression_linear(cache: ProjectCacheDependency, input: LinearRegressionInput)->ApiResult[LinearRegressionResult]:
  return ApiResult(
    data=linear_regression(cache, input),
    message=None,
  )

@router.post("/regression/logistic")
async def post__logistic_regression(cache: ProjectCacheDependency, input: LogisticRegressionInput)->ApiResult[LogisticRegressionResult]:
  return ApiResult(
    data=logistic_regression(cache, input),
    message=None,
  )

@router.post("/regression/logistic/multinomial")
async def post__multinomial_logistic_regression(cache: ProjectCacheDependency, input: MultinomialLogisticRegressionInput)->ApiResult[MultinomialLogisticRegressionResult]:
  return ApiResult(
    data=multinomial_logistic_regression(cache, input),
    message=None,
  )

@router.post("/regression/ordinal")
async def post__ordinal_regression(cache: ProjectCacheDependency, input: OrdinalRegressionInput)->ApiResult[OrdinalRegressionResult]:
  return ApiResult(
    data=ordinal_regression(cache, input),
    message=None,
  )
