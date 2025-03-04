import http
import os

from modules.api import ApiResult, ApiError
from modules.config import Config
from models.project import ProjectLiteResource, ProjectResource, UpdateProjectIdSchema
from modules.project.cache import ProjectCacheManager
from modules.project.paths import DATA_DIRECTORY, ProjectPathManager, ProjectPaths
from modules.logger.provisioner import ProvisionedLogger
from modules.task.engine import TaskEngine

from .dependency import _assert_project_id_doesnt_exist

logger = ProvisionedLogger().provision("Project Controller")

def get_all_projects():
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

def create_project(config: Config):
  # Condition checks
  _assert_project_id_doesnt_exist(config.project_id)
  os.makedirs(config.paths.project_path, exist_ok=True)

  # Create workspace
  logger.info(f"Loading dataset from {config.source.path} with type {config.source.type}")
  df = config.source.load()

  logger.info(f"Fitting dataset")
  df = config.data_schema.fit(df)

  # Commit changes
  logger.info(f"Saving configuration to {config.paths.config_path}")
  config.save_to_json()
  
  workspace_path = config.paths.full_path(ProjectPaths.Workspace)
  logger.info(f"Saving fitted dataset in {workspace_path}")
  config.save_workspace(df)

  return ApiResult(
    data=ProjectResource(
      id=config.project_id,
      config=config,
      path=config.paths.project_path
    ),
    message=f"Your new project \"{config.project_id}\", has been successfully created."
  )

def update_project_id(config: Config, body: UpdateProjectIdSchema):
  # Condition checks
  _assert_project_id_doesnt_exist(body.project_id)
  new_paths = ProjectPathManager(project_id=body.project_id)

  # Action
  os.rename(config.paths.project_path, new_paths.project_path)

  # Invalidate cache
  ProjectCacheManager().invalidate(config.project_id)
  TaskEngine().clear_tasks(config.project_id)

def update_project(old_config: Config, new_config: Config):
  if old_config.project_id != new_config.project_id:
    raise ApiError("We do not support updating the project ID in this endpoint. Consider using /api/projects/:id/update-project-id instead", http.HTTPStatus.BAD_REQUEST)
  
  if old_config.source != new_config.source:
    raise ApiError("We do not support updating the data source after a project has been created to avoid any file corruption. Consider creating a new project instead.", http.HTTPStatus.BAD_REQUEST)

  # Resolve project differences
  df = old_config.load_workspace()
  logger.info(f"Resolving differences in the configurations of \"{new_config.project_id}\"")
  df = new_config.data_schema.resolve_difference(old_config.data_schema, df)
  logger.info(f"Successfully resolved the differences in the column configurations of \"{new_config.project_id}\"")

  # Commit changes
  new_config.save_to_json()
  new_config.save_workspace(df)

  # Invalidate cache
  ProjectCacheManager().invalidate(old_config.project_id)
  TaskEngine().clear_tasks(old_config.project_id)

  return ApiResult(
    data=ProjectResource(
      id=new_config.project_id,
      config=new_config,
      path=new_config.paths.project_path
    ),
    message=f"Project \"{new_config.project_id}\" has been successfully updated. All of the previously cached results has been invalidated to account for the modified columns/dataset, so you may have to run the topic modeling procedure again."
  )

def delete_project(project_id: str):
  manager = ProjectPathManager(project_id=project_id)
  if not os.path.exists(manager.project_path):
    raise ApiError(f"We cannot find any projects with ID: \"{project_id}\", perhaps it had been manually deleted by a user?", 404)

  manager.cleanup(all=True)
  TaskEngine().clear_tasks()
  ProjectCacheManager().invalidate(project_id)

  return ApiResult(
    data=None,
    message=f"Project \"{project_id}\" has been successfully deleted."
  )

__all__ = [
  "get_all_projects",
  "create_project",
  "update_project_id",
  "update_project",
  "delete_project"
]