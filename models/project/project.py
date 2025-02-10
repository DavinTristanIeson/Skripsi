import http
import os
from typing import Annotated, Any, Hashable, Optional, Sequence

import numpy as np
import pandas as pd
import pydantic
from common.models.api import ApiError
from common.models.validators import FilenameField, CommonModelConfig

from ..config import Config, SchemaColumnTypeEnum, DataSource

# Resource
class ProjectLiteResource(pydantic.BaseModel):
  # This resource doesn't have any other fields for now, and probably for the foreseeable future.
  # But we're making it a resource anyway in case a new feature introduces a new field to this resource.
  id: str
  path: str

class ProjectResource(pydantic.BaseModel):
  id: str
  path: str
  config: Config


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
  document_lengths: Optional[InferDatasetDescriptiveStatisticsResource]
  

class CheckDatasetResource(pydantic.BaseModel):
  dataset_columns: list[str]
  preview_rows: list[dict[Hashable, Any]]
  total_rows: int
  columns: list[InferDatasetColumnResource]

# Schema
class CheckProjectIdSchema(pydantic.BaseModel):
  project_id: FilenameField

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
  model_config = CommonModelConfig
  source: DataSourceField
  column: str
  dtype: SchemaColumnTypeEnum

class UpdateProjectIdSchema(pydantic.BaseModel):
  project_id: str

__all__ = [
  "ProjectLiteResource",
  "ProjectResource",

  "InferDatasetColumnResource",
  "InferDatasetDescriptiveStatisticsResource",
  
  "CheckDatasetColumnSchema",
  "CheckDatasetSchema",
  "CheckProjectIdSchema",
  "CheckDatasetResource",
]
