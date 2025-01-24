import abc
import datetime
from enum import Enum
import json
from typing import Annotated, Any, ClassVar, Literal, Optional, Sequence, Union, cast
import re

import numpy as np
import numpy.typing as npt
import pydantic
import pandas as pd

from common.models.api import ApiError
from common.models.validators import CommonModelConfig, FilenameField, DiscriminatedUnionValidator
from common.models.enum import ExposedEnum
from .textual import TextPreprocessingConfig, TopicModelingConfig

class SchemaColumnTypeEnum(str, Enum):
  Continuous = "continuous"
  Categorical = "categorical"
  MultiCategorical = "multi-categorical"
  Geospatial = "geospatial"
  Temporal = "temporal"
  Textual = "textual"
  Unique = "unique"

ExposedEnum().register(SchemaColumnTypeEnum)

class BaseSchemaColumn(pydantic.BaseModel, abc.ABC):
  name: FilenameField
  dataset_name: Optional[str] = None

  def access(self, df: pd.DataFrame)->pd.Series:
    return df[self.dataset_name or self.name]

  @abc.abstractmethod
  def fit(self, df: pd.DataFrame):
    pass

class ContinuousSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Continuous]
  # Will never be None after fitting
  lower_bound: Optional[float] = None
  upper_bound: Optional[float] = None

  def fit(self, df):
    data = self.access(df).astype(np.float64)
    if self.lower_bound is not None:
      data[data < self.lower_bound] = self.lower_bound
    if self.upper_bound is not None:
      data[data > self.upper_bound] = self.upper_bound
    df[self.name] = data
  
class CategoricalSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Categorical]
  min_frequency: int = pydantic.Field(default=1, gt=0)
  category_order: list[str]

  def fit(self, df):
    # Remove min_frequency
    data = self.access(df).copy()
    category_frequencies: pd.Series = data.value_counts()
    filtered_category_frequencies: pd.Series = (category_frequencies[category_frequencies < self.min_frequency])
    for category in filtered_category_frequencies.index:
      data[data == category] = pd.NA
    df[self.name] = pd.Categorical(data)

class UniqueSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Unique]
  def fit(self, df):
    return

class TextualSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Textual]
  preprocessing: TextPreprocessingConfig
  topic_modeling: TopicModelingConfig

  @property
  def preprocess_column(self):
    return f"{self.name} (Preprocessed)"

  def fit(self, df):
    data = self.access(df)
    isna_mask = data.isna()
    new_data = data.astype(str)
    new_data[isna_mask] = ''
    documents = cast(Sequence[str], new_data[~isna_mask])

    preprocessed_documents = tuple(map(lambda x: ' '.join(x), self.preprocessing.preprocess(documents)))
    new_data[~isna_mask] = preprocessed_documents
    df[self.preprocess_column] = new_data
    return
  
class TemporalSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Temporal]
  min_date: Optional[datetime.datetime]
  max_date: Optional[datetime.datetime]
  datetime_format: Optional[str]

  def fit(self, df):
    kwargs = dict()
    if self.datetime_format is not None:
      kwargs["format"] = self.datetime_format
    datetime_column = pd.to_datetime(self.access(df), **kwargs)
    if self.min_date is not None:
      datetime_column[datetime_column < self.min_date] = self.min_date
    if self.max_date is not None:
      datetime_column[datetime_column > self.max_date] = self.max_date

    df[self.name] = datetime_column
  
class MultiCategoricalSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.MultiCategorical]
  min_frequency: int = pydantic.Field(default=1, gt=0)
  delimiter: str = ","
  is_json: bool = True

  def fit(self, df):
    # Remove min_frequency
    data = self.access(df).copy()

    unique_categories = dict()
    rows = []
    for row in data:
      row_categories: list[str]
      if self.is_json:
        row_categories = list(json.loads(row))
      else:
        row_categories = list(map(lambda category: category.strip(), str(row).split(self.delimiter)))
      
      for category in row_categories:
        unique_categories[category] = unique_categories.get(category, 0) + 1
      rows.append(row_categories)
    category_frequencies: pd.Series = pd.Series(unique_categories)
    filtered_category_frequencies: pd.Series = (category_frequencies[category_frequencies < self.min_frequency]) # type: ignore

    final_rows: list[str] = []
    for row in rows:
      filtered_row = list(filter(lambda category: category not in filtered_category_frequencies.index, row))
      final_rows.append(json.dumps(filtered_row))
    df[self.name] = pd.Series(rows)
  
class GeospatialSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  latitude_dataset_name: str
  longitude_dataset_name: str
  min_frequency: int = pydantic.Field(default=1, gt=0)
  merge_distance: float = 0

  @property
  def latitude_name(self):
    return f"{self.name} (Latitude)"

  @property
  def longitude_name(self):
    return f"{self.name} (Longitude)"

  # https://stackoverflow.com/questions/365826/calculate-distance-between-2-gps-coordinates
  @staticmethod
  def haversine_distance_km(a: npt.NDArray, b: npt.NDArray):
    degree_to_radian = np.pi / 180.0
    dlong = (a[1] - b[1]) * degree_to_radian
    dlat = (a[0] - b[0]) * degree_to_radian
    a = pow(np.sin(dlat/2.0), 2) + np.cos(a[0]*degree_to_radian) * np.cos(b[0]*degree_to_radian) * pow(np.sin(dlong/2.0), 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = 6367 * c

    return d

  def fit(self, df):
    coordinates = df.loc[:, [self.latitude_dataset_name, self.longitude_dataset_name]]

    # Get all unique coordinates
    unique_coordinates = coordinates.apply(lambda row: f"{row[self.latitude_dataset_name]}, {row[self.longitude_dataset_name]}", axis=1)
    coordinate_frequencies = unique_coordinates.value_counts()

    # Get infrequent unique coordinates.
    filtered_coordinate_frequencies: pd.Series = (coordinate_frequencies[coordinate_frequencies < self.min_frequency])
    unassigned_mask = np.full(len(coordinates), True)
    for serialized_coordinate in filtered_coordinate_frequencies.index:
      unassigned_mask[unique_coordinates == serialized_coordinate, :] = False
    unassigned = coordinates[unassigned_mask]

    # Try to merge infrequent coordinates
    from sklearn.cluster import DBSCAN
    dbscan_model = DBSCAN(eps=self.merge_distance, min_samples=3, metric=GeospatialSchemaColumn.haversine_distance_km)
    clusters = pd.Series(dbscan_model.fit_predict(unassigned))

    cluster_frequencies = clusters.value_counts()
    filtered_cluster_frequencies: pd.Series = (cluster_frequencies[cluster_frequencies < self.min_frequency])
    resurrected_cluster_frequencies: pd.Series = (cluster_frequencies[cluster_frequencies >= self.min_frequency])
    # Entirely remove infrequent coordinates that cannot be clustered enough
    for cluster in filtered_cluster_frequencies.index:
      cluster_mask = clusters == cluster
      unassigned[cluster_mask] = pd.NA

    # Use the centroid of frequent coordinates to represent it
    for cluster in resurrected_cluster_frequencies.index:
      cluster_mask = clusters == cluster
      if cluster == -1:
        unassigned[cluster_mask] = pd.NA
        continue
      unassigned[cluster_mask] = unassigned[cluster_mask].mean()

    # Put the resolved unassigned coordinates back to the original array
    coordinates[unassigned_mask] = unassigned

    df.loc[:, [self.latitude_name, self.longitude_name]] = coordinates

SchemaColumn = Annotated[Union[UniqueSchemaColumn, CategoricalSchemaColumn, TextualSchemaColumn, ContinuousSchemaColumn, TemporalSchemaColumn], pydantic.Field(discriminator="type"), DiscriminatedUnionValidator]

__all__ = [
  "BaseSchemaColumn",
  "SchemaColumn",
  "TextualSchemaColumn",
  "UniqueSchemaColumn",
  "CategoricalSchemaColumn",
  "ContinuousSchemaColumn",
  "SchemaColumnTypeEnum",
]