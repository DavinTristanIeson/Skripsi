import http
import os
import json
from typing import Any, Hashable, Optional, Sequence

import pandas as pd
import pydantic
from common.models.api import ApiError
from common.models.validators import FilenameField
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
    iqr = summary["q3"] - summary["q1"]
    inlier_range = (summary["q1"] - (1.5 * iqr), summary["q3"] + (1.5 * iqr))

    inlier_mask = (column >= inlier_range[0]) | (column <= inlier_range[1])
    outlier_count = summary[~inlier_mask].count()
    return InferDatasetDescriptiveStatisticsResource(
      count=summary["count"],
      mean=summary["mean"],
      median=summary["q2"],
      std=summary["std"],
      min=summary["min"],
      q1=summary["q1"],
      q3=summary["q3"],
      max=summary["max"],
      inlier_range=inlier_range,
      outlier_count=outlier_count,
    )
  
class InferDatasetCategoryFrequenciesResource(pydantic.BaseModel):
  frequency_threshold: int
  should_be_excluded: list[str]
  frequencies: dict[str, int]

  @staticmethod
  def infer_category_frequencies(category_frequencies: pd.Series[int]):
    global_frequency_threshold = 0.8 * category_frequencies.sum()
    latest_frequency_threshold: int = category_frequencies[0]
    for category, frequency in category_frequencies.items():
      global_frequency_threshold -= frequency
      latest_frequency_threshold = frequency
      if global_frequency_threshold <= 0:
        break

    return InferDatasetCategoryFrequenciesResource(
      frequency_threshold=latest_frequency_threshold,
      should_be_excluded=list(category_frequencies[category_frequencies < latest_frequency_threshold].index),
      frequencies=category_frequencies[category_frequencies < latest_frequency_threshold].to_dict(),
    )

  @staticmethod
  def from_categorical_series(column: pd.Series):
    return InferDatasetCategoryFrequenciesResource.infer_category_frequencies(column.value_counts())
  
  @staticmethod
  def from_multi_categorical_series(column: pd.Series):
    raw_category_frequencies: dict[str, int] = dict()
    for row in column:
      for category in json.loads(row):
        raw_category_frequencies[category] = raw_category_frequencies.get(category, 0) + 1
    category_frequencies: pd.Series[int] = pd.Series(category_frequencies) # type: ignore
    category_frequencies.sort_values(inplace=True)

    return InferDatasetCategoryFrequenciesResource.infer_category_frequencies(category_frequencies)
    

class InferDatasetColumnResource(pydantic.BaseModel):
  # Configurations that FE can use to autofill schema.
  name: str
  type: SchemaColumnTypeEnum

  total_rows: int
  category_frequencies: Optional[InferDatasetCategoryFrequenciesResource] = None
  descriptive_statistics: Optional[InferDatasetDescriptiveStatisticsResource] = None
  

class CheckDatasetResource(pydantic.BaseModel):
  dataset_columns: list[str]
  preview_rows: list[dict[Hashable, Any]]
  columns: list[InferDatasetColumnResource]

# Schema
class CheckProjectIdSchema(pydantic.BaseModel):
  project_id: FilenameField

class CheckDatasetSchema(pydantic.RootModel):
  root: DataSource

  @pydantic.field_validator('root', mode="after")
  def validate_file_path(cls, root: DataSource):
    if not os.path.exists(root.path):
      raise ApiError(f"Cannot find any file at {root.path}. Are you sure you have provided the correct path?", http.HTTPStatus.NOT_FOUND)
    
    if not os.path.isfile(root.path):
      raise ApiError(f"The item at {root.path} is not a file. Are you sure you have provided the correct path?", http.HTTPStatus.BAD_REQUEST)
    
    return root