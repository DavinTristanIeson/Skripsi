from fastapi import APIRouter

from controllers.project.dependency import ProjectCacheDependency
from models.correlation import TopicCorrelationCrossTabResource, TopicCorrelationFineGrainedCategoricalResource, TopicCorrelationResource, TopicCorrelationSchema
from modules.api.wrapper import ApiResult

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