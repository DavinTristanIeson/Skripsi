import http
import os
from typing import Annotated

from fastapi import Body, Depends, Query

from modules.api import ApiError
from modules.config import ProjectCacheManager, Config, ProjectCache, SchemaColumn, ProjectPathManager

def __get_cached_project(project_id: Annotated[str, Query()]):
  return ProjectCacheManager().get(project_id)

def _assert_project_id_doesnt_exist(project_id: Annotated[str, Query()]):
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
    raise ApiError(f"Column {column} doesn't exist in the schema. Please make sure that your schema is properly configured to your data.", http.HTTPStatus.NOT_FOUND)
SchemaColumnExistsDependency = Annotated[SchemaColumn, Depends(__get_data_column)]


__all__ = [
  "ProjectExistsDependency",
  "ProjectCacheDependency"
]
