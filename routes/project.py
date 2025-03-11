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
  check_if_project_exists,
  get_all_projects,
  create_project,
  update_project,
  update_project_id,
  delete_project,
)
from models.project import (
  CheckDatasetResource,
  CheckDatasetSchema,
  CheckProjectIdSchema,
  InferDatasetColumnResource,
  DatasetPreviewResource,
  ProjectLiteResource,
  ProjectResource,
  CheckDatasetColumnSchema,
  UpdateProjectIdSchema,
  UpdateProjectSchema
)
 
router = APIRouter(
  tags=['Projects'],
)

@router.post(
  "/check-project-id", 
  status_code=http.HTTPStatus.OK,
)
async def post__check_project_id(body: CheckProjectIdSchema)->ApiResult[None]:
  return check_if_project_exists(body)

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
async def get__projects()->ApiResult[list[ProjectLiteResource]]:
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
async def create__project(config: Config)->ApiResult[ProjectResource]:
  return create_project(config)
  
@router.patch('/{project_id}/update-project-id')
async def update__project_id(cache: ProjectCacheDependency, body: UpdateProjectIdSchema)->ApiResult[None]:
  return update_project_id(cache.config, body)
  
@router.put('/{project_id}')
async def update__project(cache: ProjectCacheDependency, body: UpdateProjectSchema)->ApiResult[ProjectResource]:
  config = cache.config
  return update_project(config, body)

@router.delete('/{project_id}')
async def delete__project(project_id: str)->ApiResult[None]:
  return delete_project(project_id)