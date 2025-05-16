from fastapi import APIRouter

from modules.api.wrapper import ApiResult
from modules.comparison.engine import TableComparisonResult

from routes.dependencies.project import ProjectCacheDependency

from .controller import compare_group_words, statistic_test, subdataset_cooccurrence
from .model import CompareSubdatasetsSchema, ComparisonStatisticTestSchema, SubdatasetCooccurrenceResource
from ..table.model import TableTopicsResource

router = APIRouter(
  tags=['Comparison']
)

@router.post("/statistic-test")
async def post__statistic_test(body: ComparisonStatisticTestSchema, cache: ProjectCacheDependency)->ApiResult[TableComparisonResult]:
  return statistic_test(body, cache)

@router.post("/words")
async def post__compare_group_words(body: CompareSubdatasetsSchema, cache: ProjectCacheDependency)->ApiResult[TableTopicsResource]:
  return compare_group_words(body, cache)
  
@router.post("/co-occurrence")
async def post__cooccurrence(body: CompareSubdatasetsSchema, cache: ProjectCacheDependency)->ApiResult[SubdatasetCooccurrenceResource]:
  return ApiResult(
    data=subdataset_cooccurrence(body, cache),
    message=None,
  )
  
