from fastapi import APIRouter

from modules.api.wrapper import ApiResult

from controllers.project import ProjectCacheDependency
from routes.correlation.controller import contingency_table

from .model import ContingencyTableResource, FineGrainedStatisticTestOnCategoriesResource, StatisticTestOnDistributionResource, TopicCorrelationSchema

router = APIRouter(
  tags=['Correlation']
)

@router.post("/topics/correlation")
def post__topics_correlation(body: TopicCorrelationSchema, cache: ProjectCacheDependency)->ApiResult[StatisticTestOnDistributionResource]:
  ...

@router.post("/topics/contingency-table")
def post__topics_contingency_table(body: TopicCorrelationSchema, cache: ProjectCacheDependency)->ApiResult[ContingencyTableResource]:
  return ApiResult(data=contingency_table(cache, body), message=None)
  
@router.post("/topics/categorical")
def post__topics_categorical_correlation(body: TopicCorrelationSchema, cache: ProjectCacheDependency)->ApiResult[FineGrainedStatisticTestOnCategoriesResource]:
  ...