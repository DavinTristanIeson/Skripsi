from fastapi import APIRouter

from modules.api.wrapper import ApiResult

from controllers.project import ProjectCacheDependency

from .model import TopicCorrelationCrossTabResource, TopicCorrelationFineGrainedCategoricalResource, TopicCorrelationResource, TopicCorrelationSchema

router = APIRouter(
  tags=['Correlation']
)

@router.post("/topics/correlation")
def post__topics_correlation(body: TopicCorrelationSchema, cache: ProjectCacheDependency)->ApiResult[TopicCorrelationResource]:
  ...

@router.post("/topics/crosstab")
def post__topics_crosstab(body: TopicCorrelationSchema, cache: ProjectCacheDependency)->ApiResult[TopicCorrelationCrossTabResource]:
  ...
  
@router.post("/topics/categorical")
def post__topics_categorical_correlation(body: TopicCorrelationSchema, cache: ProjectCacheDependency)->ApiResult[TopicCorrelationFineGrainedCategoricalResource]:
  ...