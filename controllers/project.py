import http
import os
from typing import Annotated, Optional

from fastapi import Depends, Query
import pandas as pd
from common.logger import RegisteredLogger
from common.models.api import ApiError

from models.config import Config, SchemaColumn, SchemaColumnTypeEnum, ProjectPathManager, ProjectPaths
from models.project import InferDatasetColumnResource, InferDatasetDescriptiveStatisticsResource, ProjectCacheDependency, ProjectCacheManager


def get_data_column(cache: ProjectCacheDependency, column: str = Query()):
  try:
    return cache.config.data_schema.assert_exists(column)
  except KeyError:
    raise ApiError(f"Column {column} doesn't exist in the schema. Please make sure that your schema is properly configured to your data.", 404)
SchemaColumnExistsDependency = Annotated[SchemaColumn, Depends(get_data_column)]

logger = RegisteredLogger().provision("Project Controller")

def infer_column_by_type(column: str, df: pd.DataFrame, dtype: SchemaColumnTypeEnum):
  data = df[column]
  document_lengths: Optional[InferDatasetDescriptiveStatisticsResource] = None
  categories: Optional[list[str]] = None
  if dtype == SchemaColumnTypeEnum.Textual:
    data = data.astype(str)
    document_lengths = InferDatasetDescriptiveStatisticsResource.from_series(data.str.len())
  elif dtype == SchemaColumnTypeEnum.Categorical:
    data = data.astype(str)
    categories = sorted(set(map(str, data.unique())))

  logger.info(f"Inferred type {dtype} for column {column}.")
  return InferDatasetColumnResource(
    name=column,
    type=dtype,
    document_lengths=document_lengths,
    categories=categories,
    count=data.count(),
  )

def infer_column_without_type(column: str, df: pd.DataFrame)->InferDatasetColumnResource:
  data = df[column]
  dtype = data.dtype
  logger.info(f"Inferring autofill values for column {column}")
  if data.count() == 0:
    logger.warning(f"Column {column} is empty.")
    return InferDatasetColumnResource(
      name=column,
      type=SchemaColumnTypeEnum.Unique,
      document_lengths=None,
      categories=None,
      count=0,
    )
  try:
    if pd.api.types.is_numeric_dtype(dtype):
      if column.lower().startswith("lat") or column.lower().startswith("long"):
        return infer_column_by_type(column, df, SchemaColumnTypeEnum.Geospatial)
      else:
        return infer_column_by_type(column, df, SchemaColumnTypeEnum.Continuous)
    else:
      uniquescnt = len(data.unique())
      if uniquescnt < 0.1 * data.count():
        return infer_column_by_type(column, df, SchemaColumnTypeEnum.Categorical)
      else:
        is_string = pd.api.types.is_string_dtype(dtype)
        has_long_text = is_string and data.str.len().mean() >= 20

        if has_long_text:
          return infer_column_by_type(column, df, SchemaColumnTypeEnum.Textual)
        else:
          return infer_column_by_type(column, df, SchemaColumnTypeEnum.Unique)
  except Exception as e:
    logger.error(e)
  logger.warning(f"Unable to infer any autofill values for column {column} as it doesn't fulfill any of the conditions.")
  return infer_column_by_type(column, df, SchemaColumnTypeEnum.Unique)

def assert_project_id_doesnt_exist(project_id: str):
  paths = ProjectPathManager(project_id=project_id)
  if os.path.exists(paths.project_path):
    raise ApiError(f"Project \"{project_id}\" already exists. Please try another name.", http.HTTPStatus.UNPROCESSABLE_ENTITY)

# This is not cached
ProjectExistsDependency = Annotated[Config, Depends(Config.from_project)]

__all__ = [
  "SchemaColumnExistsDependency",
  "infer_column_without_type",
  "assert_project_id_doesnt_exist",
  "ProjectExistsDependency"
]