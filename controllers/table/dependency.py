
import http
from typing import Annotated
from fastapi import Body, Depends, Query

from modules.api import ApiError
from modules.config import SchemaColumn

from controllers.project.dependency import ProjectCacheDependency

def __get_data_column(cache: ProjectCacheDependency, column: Annotated[str, Body()]):
  try:
    return cache.config.data_schema.assert_exists(column)
  except KeyError:
    raise ApiError(f"Column {column} doesn't exist in the schema. Please make sure that your schema is properly configured to your data.", http.HTTPStatus.NOT_FOUND)
SchemaColumnExistsDependency = Annotated[SchemaColumn, Depends(__get_data_column)]
