from fastapi import APIRouter

from controllers.comparison import compare_group_words, statistic_test
from controllers.project.dependency import ProjectCacheDependency
from models.comparison import ComparisonGroupWordsSchema, ComparisonStatisticTestSchema
from models.table import TableTopicsResource
from modules.api.wrapper import ApiResult
from modules.comparison.engine import TableComparisonResult


router = APIRouter(
  tags=['Comparison']
)

@router.post("/statistic-test")
async def post__statistic_test(body: ComparisonStatisticTestSchema, cache: ProjectCacheDependency)->ApiResult[TableComparisonResult]:
  return statistic_test(body, cache)

@router.post("/words")
async def post__compare_group_words(body: ComparisonGroupWordsSchema, cache: ProjectCacheDependency)->ApiResult[TableTopicsResource]:
  return compare_group_words(body, cache)
  
