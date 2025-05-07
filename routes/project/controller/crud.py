import os

from modules.api import ApiResult
from modules.config import Config
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.exceptions.dataframe import DataFrameLoadException
from modules.exceptions.files import FileLoadingException, FileNotExistsException
from modules.project.cache import ProjectCache, ProjectCacheManager, get_cached_data_source
from modules.project.paths import DATA_DIRECTORY, ProjectPathManager, ProjectPaths
from modules.logger.provisioner import ProvisionedLogger
from modules.task.engine import scheduler, topic_modeling_job_store
from modules.task.storage import TaskStorage

from ..model import ProjectMutationSchema, ProjectResource
from .project_checks import _assert_valid_project_id

from controllers.project import _assert_project_id_doesnt_exist

logger = ProvisionedLogger().provision("Project Controller")

def get_all_projects():
  folder_name = os.path.join(os.getcwd(), DATA_DIRECTORY)
  projects: list[ProjectResource] = []

  if os.path.isdir(folder_name):
    folders = [
      name for name in os.listdir(folder_name)
      if os.path.isdir(os.path.join(folder_name, name))
      # has config.json
      and os.path.exists(os.path.join(folder_name, name, ProjectPaths.Config))
      # don't include hidden folders
      and not folder_name.startswith('.')
    ]

    for folder in folders:
      # Cache the configs
      cache = ProjectCacheManager().get(folder)
      try:
        projects.append(ProjectResource.from_config(cache.config))
      except FileLoadingException as e:
        continue

  return ApiResult(
    data=projects,
    message=None
  )

def create_project(body: ProjectMutationSchema):
  import uuid
  paths = ProjectPathManager(project_id=uuid.uuid4().hex)
  # Condition checks
  _assert_project_id_doesnt_exist(paths.project_id)
  _assert_valid_project_id(paths.project_id)
  os.makedirs(paths.project_path, exist_ok=True)

  # Create workspace
  logger.info(f"Loading dataset from {body.source.path} with type {body.source.type}")
  df = body.source.load()

  logger.info(f"Fitting dataset")
  df = body.data_schema.fit(df)

  # Commit changes
  logger.info(f"Saving configuration to {paths.config_path}")

  config = Config(
    data_schema=body.data_schema,
    project_id=paths.project_id,
    metadata=body.metadata,
    source=body.source,
    version=1,
  )
  config.save_to_json()
  
  workspace_path = config.paths.full_path(ProjectPaths.Workspace)
  logger.info(f"Saving fitted dataset in {workspace_path}")
  config.save_workspace(df)

  return ApiResult(
    data=ProjectResource.from_config(config),
    message=f"Your new project \"{body.metadata.name}\", has been successfully created."
  )

def update_project(config: Config, body: ProjectMutationSchema):
  # Resolve project differences

  try:
    workspace_df = config.load_workspace()
  except (DataFrameLoadException, FileNotExistsException):
    workspace_df = None

  logger.info(f"Resolving differences in the configurations of \"{config.project_id}\"")

  new_config = Config(
    data_schema=body.data_schema,
    project_id=config.project_id,
    metadata=body.metadata,
    source=body.source,
    version=1,
  )
  df, column_diffs = new_config.data_schema.resolve_difference(
    prev=config.data_schema,
    workspace_df=workspace_df,
    source_df=get_cached_data_source(new_config.source)
  )
  cleanup_targets: list[str] = []
  for diff in column_diffs:
    if (diff.current is None or diff.current.type != diff.previous.type) and diff.previous.type == SchemaColumnTypeEnum.Textual:
      cleanup_targets.append(ProjectPaths.TopicModelingFolder(diff.previous.name))
  logger.info(f"Successfully resolved the differences in the column configurations of \"{config.project_id}\"")

  # Commit changes
  new_config.save_to_json()
  new_config.save_workspace(df)

  # Invalidate cache
  ProjectCacheManager().invalidate(new_config.project_id)
  TaskStorage().invalidate(prefix=new_config.project_id, clear=True)
  new_config.paths._cleanup([], cleanup_targets)

  return ApiResult(
    data=ProjectResource(
      id=new_config.project_id,
      config=new_config,
      path=new_config.paths.project_path
    ),
    message=f"Project \"{new_config.metadata.name}\" has been successfully updated. All of the previously cached results has been invalidated to account for the modified columns/dataset, so you may have to run the topic modeling procedure again."
  )

def delete_project(config: Config):
  config.paths.cleanup(all=True)
  scheduler.remove_all_jobs(topic_modeling_job_store)
  ProjectCacheManager().invalidate(config.project_id)
  TaskStorage().invalidate(prefix=config.project_id, clear=True)

  return ApiResult(
    data=None,
    message=f"Project \"{config.metadata.name}\" has been successfully deleted."
  )

def reload_project(cache: ProjectCache):
  config = cache.config
  df = get_cached_data_source(config.source)

  config.paths.cleanup()
  scheduler.remove_all_jobs(topic_modeling_job_store)
  ProjectCacheManager().invalidate(config.project_id)  

  df = config.data_schema.fit(df)
  cache.workspaces.save(df)

  return ApiResult(
    data=None,
    message=f"The dataset has been successfully reloaded from \"{config.source.path}\". All cached data has been removed; this includes the topic modeling results."
  )

__all__ = [
  "get_all_projects",
  "create_project",
  "update_project",
  "delete_project",
  "reload_project",
]