import http
import os
from typing import Annotated, Any, Hashable, Optional, Sequence

import numpy as np
import pandas as pd
import pydantic

from modules.api import ApiError
from modules.config.config import ProjectMetadata
from modules.config.schema.schema_manager import SchemaManager
from modules.config import Config, SchemaColumnTypeEnum, DataSource

# Resource
class ProjectResource(pydantic.BaseModel):
  id: str
  path: str
  config: Config

  @staticmethod
  def from_config(config: Config)->"ProjectResource":
    return ProjectResource(
      id=config.project_id,
      config=config,
      path=config.paths.project_path
    )


class InferDatasetDescriptiveStatisticsResource(pydantic.BaseModel):
  count: float
  mean: float
  median: float
  std: float
  min: float
  q1: float
  q3: float
  max: float
  inlier_range: tuple[float, float]
  outlier_count: int

  @staticmethod
  def from_series(column: pd.Series):
    summary = column.describe()
    iqr = summary["75%"] - summary["25%"]
    inlier_range = (summary["25%"] - (1.5 * iqr), summary["75%"] + (1.5 * iqr))

    inlier_mask = np.bitwise_or(column >= inlier_range[0], column <= inlier_range[1])
    outlier_count = column[~inlier_mask].count()
    return InferDatasetDescriptiveStatisticsResource(
      count=summary["count"],
      mean=summary["mean"],
      median=summary["50%"],
      std=summary["std"],
      min=summary["min"],
      q1=summary["25%"],
      q3=summary["75%"],
      max=summary["max"],
      inlier_range=inlier_range,
      outlier_count=outlier_count,
    )

class InferDatasetColumnResource(pydantic.BaseModel):
  # Configurations that FE can use to autofill schema.
  name: str
  type: SchemaColumnTypeEnum

  count: int
  categories: Optional[list[str]]
  descriptive_statistics: Optional[InferDatasetDescriptiveStatisticsResource]
  

class CheckDatasetResource(pydantic.BaseModel):
  columns: list[InferDatasetColumnResource]

class DatasetPreviewResource(pydantic.BaseModel):
  dataset_columns: list[str]
  preview_rows: list[dict[str, Any]]
  total_rows: int

# Schema
class CheckProjectIdSchema(pydantic.BaseModel):
  project_id: str

def validate_file_path(data: DataSource):
  if not os.path.exists(data.path):
    raise ApiError(f"Cannot find any file at {data.path}. Are you sure you have provided the correct path?", http.HTTPStatus.NOT_FOUND)
  if not os.path.isfile(data.path):
    raise ApiError(f"The item at {data.path} is not a file. Are you sure you have provided the correct path?", http.HTTPStatus.BAD_REQUEST)
  return data
DataSourceField = Annotated[DataSource, pydantic.AfterValidator(validate_file_path)]

class CheckDatasetSchema(pydantic.RootModel):
  root: DataSourceField

class CheckDatasetColumnSchema(pydantic.BaseModel):
  source: DataSourceField
  column: str
  dtype: SchemaColumnTypeEnum


class ProjectMutationSchema(pydantic.BaseModel):
  metadata: ProjectMetadata
  source: DataSourceField
  data_schema: SchemaManager


__all__ = [
  "ProjectResource",

  "InferDatasetColumnResource",
  "InferDatasetDescriptiveStatisticsResource",
  
  "CheckDatasetColumnSchema",
  "CheckDatasetSchema",
  "CheckProjectIdSchema",
  "CheckDatasetResource",
]
