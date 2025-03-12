import http
from fastapi import APIRouter, BackgroundTasks

from controllers.project.crud import get_all_projects
from controllers.project.dependency import ProjectCacheDependency
from controllers.project.infer_column import get_dataset_preview
from modules.api.wrapper import ApiResult
from modules.config.config import Config
from modules.logger import ProvisionedLogger

from controllers.project import (
  infer_column_from_dataset,
  infer_columns_from_dataset,
  get_all_projects,
  create_project,
  update_project,
  delete_project,
)
from models.project import (
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
  config = cache.config
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
  config = cache.config
  return update_project(config, body)

@router.delete('/{project_id}')
async def delete__project(project_id: str)->ApiResult[None]:
  return delete_project(project_id)