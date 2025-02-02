import abc
import datetime
from enum import Enum
import json
from typing import Annotated, Literal, Optional, Sequence, Union, cast

import numpy as np
import pydantic
import pandas as pd

from common.models.validators import CommonModelConfig, FilenameField, DiscriminatedUnionValidator, validate_http_url
from common.models.enum import ExposedEnum
from .textual import TextPreprocessingConfig, TopicModelingConfig

class SchemaColumnTypeEnum(str, Enum):
  Continuous = "continuous"
  Categorical = "categorical"
  MultiCategorical = "multi-categorical"
  Temporal = "temporal"
  Textual = "textual"
  Unique = "unique"
  Geospatial = "geospatial"
  Image = "image"

ExposedEnum().register(SchemaColumnTypeEnum)

class GeospatialRoleEnum(str, Enum):
  Latitude = "latitude"
  Longitude = "longitude"

ExposedEnum().register(GeospatialRoleEnum)

class BaseSchemaColumn(pydantic.BaseModel, abc.ABC):
  name: str
  active: bool = True

  def fit(self, data: pd.Series)->pd.Series:
    return data

class ContinuousSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Continuous]

  def fit(self, data):
    data = data.astype(np.float64)
    return data
  
class CategoricalSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Categorical]
  category_order: Optional[list[str]] = None
  def fit(self, data):
    if self.category_order is None:
      new_data = pd.Categorical(data)
    else:
      new_data = pd.Categorical(
        data,
        ordered=True,
        categories=self.category_order
      )
    return cast(pd.Series, new_data)

class TextualSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  name: FilenameField
  type: Literal[SchemaColumnTypeEnum.Textual]
  preprocessing: TextPreprocessingConfig
  topic_modeling: TopicModelingConfig

  @property
  def preprocess_column(self):
    return f"{self.name} (Preprocessed)"
  
  def preprocess(self, data: pd.Series):
    isna_mask = data.isna()
    new_data = data.astype(str)
    new_data[isna_mask] = ''
    documents = cast(Sequence[str], new_data[~isna_mask])

    preprocessed_documents = tuple(map(lambda x: ' '.join(x), self.preprocessing.preprocess(documents)))
    new_data[~isna_mask] = preprocessed_documents
    return new_data

class TemporalSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Temporal]
  datetime_format: Optional[str]

  def fit(self, data):
    kwargs = dict()
    if self.datetime_format is not None:
      kwargs["format"] = self.datetime_format
    datetime_column = pd.to_datetime(data, **kwargs)
    return datetime_column
  
class MultiCategoricalSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.MultiCategorical]
  min_frequency: int = pydantic.Field(default=1, gt=0)
  delimiter: str = ","
  is_json: bool = True

  def fit(self, data):
    # Remove min_frequency
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
    return pd.Series(rows)

class GeospatialSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Geospatial]
  role: GeospatialRoleEnum
  def fit(self, data):
    # Latitude range is [-90, 90]. Longtitude range is [-180, 180]
    data = data.astype(np.float64)
    if self.role == GeospatialRoleEnum.Latitude:
      latitude_mask = np.bitwise_or(data < -90, data > 90)
      data[latitude_mask] = pd.NA
    else:
      longitude_mask = np.bitwise_or(data < -180, data > 180)
      data[longitude_mask] = pd.NA
    return data

class UniqueSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Unique]
  
class ImageSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Image]
  
  def fit(self, data):
    valid_link_mask = data.apply(validate_http_url)
    data[valid_link_mask] = pd.NA
    return data
  
SchemaColumn = Annotated[
  Union[
    UniqueSchemaColumn,
    CategoricalSchemaColumn,
    TextualSchemaColumn,
    ContinuousSchemaColumn,
    TemporalSchemaColumn,
    GeospatialSchemaColumn,
    UniqueSchemaColumn,
    ImageSchemaColumn,
  ],
  pydantic.Field(discriminator="type"),
  DiscriminatedUnionValidator
]

__all__ = [
  "BaseSchemaColumn",
  "SchemaColumn",
  "TextualSchemaColumn",
  "UniqueSchemaColumn",
  "ImageSchemaColumn",
  "CategoricalSchemaColumn",
  "MultiCategoricalSchemaColumn",
  "ContinuousSchemaColumn",
  "GeospatialSchemaColumn",
  "SchemaColumnTypeEnum",
]