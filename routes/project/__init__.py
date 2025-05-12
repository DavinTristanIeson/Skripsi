from fastapi import APIRouter

from routes.dependencies.project import ProjectCacheDependency, ProjectExistsDependency, ProjectLockDependency
from modules.api.wrapper import ApiResult
from modules.project.lock import ProjectThreadLockManager

from .controller import (
  infer_column_from_dataset,
  infer_columns_from_dataset,
  get_all_projects,
  reload_project,
  create_project,
  update_project,
  delete_project,
  get_dataset_preview
)
from .model import (
  CheckDatasetResource,
  CheckDatasetSchema,
  ProjectMutationSchema,
  InferDatasetColumnResource,
  DatasetPreviewResource,
  ProjectResource,
  CheckDatasetColumnSchema,
)
 
router = APIRouter(
  tags=['Projects'],
)

@router.post("/check-dataset")
async def post__check_dataset(body: CheckDatasetSchema)->ApiResult[CheckDatasetResource]:
  return infer_columns_from_dataset(body)
  
@router.post("/check-dataset-column")
async def check_dataset_column(body: CheckDatasetColumnSchema)->ApiResult[InferDatasetColumnResource]:
  return infer_column_from_dataset(body)

@router.post("/dataset_preview")
async def get__dataset_preview(body: CheckDatasetSchema)->ApiResult[DatasetPreviewResource]:
  return get_dataset_preview(body)


@router.get('/')
async def get__projects()->ApiResult[list[ProjectResource]]:
  return get_all_projects()

@router.get('/{project_id}')
async def get__project(cache: ProjectCacheDependency)->ApiResult[ProjectResource]:
  return ApiResult(
    data=ProjectResource(
      id=cache.id,
      config=cache.config,
      path=cache.config.paths.project_path,
    ),
    message=None
  )

@router.post('/')
async def create__project(body: ProjectMutationSchema)->ApiResult[ProjectResource]:
  return create_project(body)

@router.put('/{project_id}')
async def update__project(cache: ProjectCacheDependency, body: ProjectMutationSchema)->ApiResult[ProjectResource]:
  # WARN DATA RACE
  return update_project(cache, body)

@router.patch('/{project_id}/reload')
async def reload__project(cache: ProjectCacheDependency)->ApiResult[None]:
  # WARN DATA RACE
  return reload_project(cache)

@router.delete('/{project_id}')
async def delete__project(config: ProjectExistsDependency, lock: ProjectLockDependency)->ApiResult[None]:
  # WARN DATA RACE
  with lock:
    return delete_project(config)