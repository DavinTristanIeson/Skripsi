import http
import os
from fastapi import APIRouter
from common.models.api import ApiResult, ApiError
from common.task.server import TaskServer
from common.logger import RegisteredLogger
import controllers
from models.project import (
  ProjectCacheManager,
  get_cached_data_source,
  ProjectCacheDependency, 
  CheckDatasetResource,
  CheckDatasetSchema,
  CheckProjectIdSchema,
  InferDatasetColumnResource,
  ProjectLiteResource,
  ProjectResource,
  CheckDatasetColumnSchema
)
from models.config import DATA_DIRECTORY, Config, ProjectPathManager, ProjectPaths
 
router = APIRouter(
  tags=['Projects']
)

logger = RegisteredLogger().provision("Project Controller")

@router.post(
  "/check-project-id", 
  status_code=http.HTTPStatus.OK, 
)
async def check_project(body: CheckProjectIdSchema):
  folder_name = DATA_DIRECTORY
  folder_path = os.path.join(os.getcwd(), folder_name, body.project_id)
  if os.path.isdir(folder_path):
    available = False
    message = f"The project name \"{body.project_id}\" is already taken. Please choose a different name."
  else:
    available = True
    message = f"The project name \"{body.project_id}\" is available. You're good to go!"
  
  return ApiResult(data={"available": available}, message=f"{message}")

@router.post("/check-dataset")
async def check_dataset(body: CheckDatasetSchema):
  df = get_cached_data_source(body.root)
  columns: list[InferDatasetColumnResource] = []
  for column in df.columns:
    inferred = controllers.project.infer_column_without_type(column, df)
    columns.append(inferred)

  return ApiResult(
    data=CheckDatasetResource(
      columns=columns,
      dataset_columns=list(df.columns),
      total_rows=len(df),
      preview_rows=df.head().to_dict(orient="records")
    ),
    message=f"We have inferred the columns from the dataset at {body.root.path}. Next, please configure how you would like to process the individual columns."
  )

@router.post("/check-dataset-column")
async def check_dataset_column(body: CheckDatasetColumnSchema):
  df = get_cached_data_source(body.source)
  inferred: InferDatasetColumnResource = controllers.project.infer_column_by_type(body.column, df, body.dtype)

  return ApiResult(
    data=inferred,
    message=None
  )


@router.get('/')
async def get__projects():
  folder_name = os.path.join(os.getcwd(), DATA_DIRECTORY)
  projects: list[ProjectLiteResource] = []

  if os.path.isdir(folder_name):
    folders = [
      name for name in os.listdir(folder_name)
      if os.path.isdir(os.path.join(folder_name, name))
      # don't include hidden folders
      and not folder_name.startswith('.')
    ]
    projects = list(map(
      lambda folder: ProjectLiteResource(id=folder, path=os.path.join(folder_name, folder)),
      folders
    ))

  return ApiResult(
    data=projects,
    message=None
  )

@router.get('/{project_id}')
async def get__project(cache: ProjectCacheDependency):
  return ApiResult(
    data=ProjectResource(
      id=cache.id,
      config=cache.config,
      path=cache.config.paths.project_path,
    ),
    message=None
  )

@router.post('/')
async def create__project(config: Config):
  folder_path = os.path.join(os.getcwd(), DATA_DIRECTORY, config.project_id)
  os.makedirs(DATA_DIRECTORY, exist_ok=True)

  if os.path.isdir(folder_path):
    raise ApiError(f"Project \"{config.project_id}\" already exists. Try another name.", http.HTTPStatus.UNPROCESSABLE_ENTITY)

  os.makedirs(folder_path)

  logger.info(f"Saving configuration to {config.paths.config_path}")
  config.save_to_json()

  logger.info(f"Loading dataset from {config.source.path} with type {config.source.type}")
  df = config.source.load()

  logger.info(f"Fitting dataset")
  df = config.data_schema.fit(df)
  
  workspace_path = config.paths.full_path(ProjectPaths.Workspace)
  logger.info(f"Saving fitted dataset in {workspace_path}")
  df.to_parquet(workspace_path)

  return ApiResult(
    data=ProjectResource(
      id=config.project_id,
      config=config,
      path=Config.paths.project_path
    ),
    message=f"Your new project \"{config.project_id}\", has been successfully created."
  )

@router.put('/{project_id}')
async def update__project(cache: ProjectCacheDependency, config: Config):
  old_config = cache.config
  if config.project_id != old_config.project_id:
    if os.path.isdir(config.paths.project_path):
      raise ApiError(f"Project '{config.project_id}' already exists. Try another name.", http.HTTPStatus.UNPROCESSABLE_ENTITY)
    
    os.rename(old_config.paths.project_path, config.paths.project_path)

  config.paths.cleanup()
  TaskServer().clear_tasks(old_config.project_id)
  ProjectCacheManager().invalidate(cache.id)
  config.save_to_json()

  return ApiResult(
    data=ProjectResource(
      id=config.project_id,
      config=config,
      path=Config.paths.project_path
    ),
    message=f"Project \"{config.project_id}\" has been successfully updated. All of the previously cached results has been invalidated to account for the modified columns/dataset, so you may have to run the topic modeling procedure again."
  )

@router.delete('/{project_id}')
async def delete__project(project_id: str):
  manager = ProjectPathManager(project_id=project_id)

  if not os.path.exists(manager.project_path):
    raise ApiError(f"We cannot find any projects with ID: \"{project_id}\", perhaps it had been manually deleted by a user?", 404)

  manager.cleanup(all=True)
  TaskServer().clear_tasks()
  ProjectCacheManager().invalidate(project_id)

  return ApiResult(
    data=None,
    message=f"Project \"{project_id}\" has been successfully deleted."
  )
