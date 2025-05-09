import http
import os
import threading
from typing import Annotated

from fastapi import Body, Depends, Path

from modules.api import ApiError
from modules.config import Config, SchemaColumn
from modules.exceptions.dataframe import MissingColumnException
from modules.project.cache import ProjectCache
from modules.project.cache_manager import ProjectCacheManager
from modules.project.lock import ProjectLockManager
from modules.project.paths import ProjectPathManager

def __get_cached_project(project_id: Annotated[str, Path()]):
  return ProjectCacheManager().get(project_id)

def _assert_project_id_doesnt_exist(project_id: Annotated[str, Path()]):
  paths = ProjectPathManager(project_id=project_id)
  if os.path.exists(paths.project_path):
    raise ApiError(f"Project \"{project_id}\" already exists. Please try another name.", http.HTTPStatus.UNPROCESSABLE_ENTITY)
  return project_id

ProjectDoesntExistDependency = Annotated[str, Depends(_assert_project_id_doesnt_exist)]

# Used for non-cached project checks
ProjectExistsDependency = Annotated[Config, Depends(Config.from_project)]

# For just getting project cache quickly
ProjectCacheDependency = Annotated[ProjectCache, Depends(__get_cached_project)]

def __get_data_column(cache: ProjectCacheDependency, column: Annotated[str, Body()]):
  try:
    return cache.config.data_schema.assert_exists(column)
  except KeyError:
    raise MissingColumnException(
      message=MissingColumnException.format_schema_issue(column)
    )
SchemaColumnExistsDependency = Annotated[SchemaColumn, Depends(__get_data_column)]

def __get_project_lock(project_id: str):
  return ProjectLockManager().get(project_id)

ProjectLockDependency = Annotated[threading.RLock, Depends(__get_project_lock)]

__all__ = [
  "ProjectExistsDependency",
  "ProjectCacheDependency"
]
