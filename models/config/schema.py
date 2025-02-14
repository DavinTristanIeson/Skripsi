import abc
from enum import Enum
import functools
import math
from typing import Annotated, Literal, Optional, Sequence, Union, cast

import numpy as np
import pydantic
import pandas as pd

from common.constants import DAYS_OF_WEEK, HOURS, MONTHS
from common.models.validators import CommonModelConfig, FilenameField, DiscriminatedUnionValidator
from common.models.enum import ExposedEnum
from models.topic.topic import TopicModelingResultModel
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

  # Internal
  Topic = "topic"

ExposedEnum().register(SchemaColumnTypeEnum)

class GeospatialRoleEnum(str, Enum):
  Latitude = "latitude"
  Longitude = "longitude"

ExposedEnum().register(GeospatialRoleEnum)

class BaseSchemaColumn(pydantic.BaseModel, abc.ABC):
  name: str
  internal: bool = pydantic.Field(default=False, exclude=True)
  active: bool = True

  def get_internal_columns(self)->Sequence["SchemaColumn"]:
    return []

  @abc.abstractmethod
  def fit(self, df: pd.DataFrame)->None:
    ...

class ContinuousSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Continuous]
  bins: Optional[int] = None

  @functools.cached_property
  def bins_column(self):
    return CategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.Categorical,
      name=f"{self.name} (Bins)",
      internal=True,
      alphanumeric_order=True
    )
  
  def get_internal_columns(self):
    return [self.bins_column]
  
  def __format_3f(self, value: float)->str:
    return f"{value:.3f}".rstrip('0').rstrip('.')

  def fit(self, df):
    data = df[self.name].astype(np.float64)
    df[self.name] = data

    if self.bins is None:
      histogram_edges = np.histogram_bin_edges(data, "auto")
    else:
      histogram_edges = np.histogram_bin_edges(data, self.bins)
    bins = np.digitize(data, histogram_edges)
    digit_length = math.floor(math.log10(len(histogram_edges))) + 1
    categorical_bins = pd.Categorical(bins, categories=np.arange(1, len(histogram_edges)+1), ordered=True)

    bin_categories = dict()
    for category in range(1, len(histogram_edges)):
      current_histogram_edge_str = self.__format_3f(histogram_edges[category])
      prev_histogram_edge_str = self.__format_3f(histogram_edges[category - 1])
      hist_range = f"[{prev_histogram_edge_str}, {current_histogram_edge_str})"

      bin_name = str(category + 1).rjust(digit_length, '0')
      bin_categories[category] = f"BIN{bin_name}: {hist_range}"
    
    first_histogram_edge_str = self.__format_3f(histogram_edges[0])
    first_bin_id = str(1).rjust(digit_length, '0') 
    bin_categories[0] = f"BIN{first_bin_id}: (-inf, {first_histogram_edge_str})"

    last_histogram_edge_str = self.__format_3f(histogram_edges[len(histogram_edges) - 1])
    last_bin_id = str(len(histogram_edges) + 1).rjust(digit_length, '0')
    bin_categories[len(histogram_edges)] = f"BIN{last_bin_id}: [{last_histogram_edge_str}, inf)"
    
    categorical_bins = categorical_bins.rename_categories(bin_categories)
    categorical_bins = categorical_bins.set_categories(sorted(bin_categories.values()), ordered=True)

    df[self.bins_column.name] = categorical_bins
  
class CategoricalSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Categorical]
  category_order: Optional[list[str]] = None
  alphanumeric_order: bool = False
  def fit(self, df):
    raw_data = df[self.name]
    if raw_data.dtype == 'category':
      data = pd.Categorical(raw_data)
    else:  
      data = pd.Categorical(raw_data.astype(str))

    category_order: Optional[list[str]] = None  
    if self.category_order is not None:
      category_order = self.category_order
    elif self.alphanumeric_order:
      category_order = sorted(data.categories)

  
    if category_order is not None:
      invalid_categories_in_config = set(category_order).difference(data.categories)
      missing_categories_in_config = set(data.categories).difference(category_order)
      
      # Ensure that category order is proper
      resolved_category_order: list[str] = sorted(missing_categories_in_config)
      for category in category_order:
        if category in invalid_categories_in_config:
          continue
        resolved_category_order.append(category)

      data = data.reorder_categories(
        resolved_category_order,
        ordered=True
      )
    df[self.name] = data

class TopicSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  type: Literal[SchemaColumnTypeEnum.Topic]
  def fit(self, df):
    pass

class TextualSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  name: FilenameField
  type: Literal[SchemaColumnTypeEnum.Textual]
  preprocessing: TextPreprocessingConfig
  topic_modeling: TopicModelingConfig

  @functools.cached_property
  def preprocess_column(self):
    return UniqueSchemaColumn(
      type=SchemaColumnTypeEnum.Unique,
      name=f"{self.name} (Preprocessed)",
      internal=True
    )

  @functools.cached_property
  def topic_column(self):
    return TopicSchemaColumn(
      type=SchemaColumnTypeEnum.Topic,
      name=f"{self.name} (Topic)",
      internal=True
    )
  
  def get_internal_columns(self):
    return [
      self.preprocess_column,
      self.topic_column,
    ]
  
  def fit(self, df):
    data = df[self.name].astype(str)
    df[self.name] = data


class TemporalSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Temporal]
  datetime_format: Optional[str]

  @functools.cached_property
  def year_column(self):
    return CategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.Categorical,
      alphanumeric_order=True,
      name=f"{self.name} (Year)",
      internal=True
    )
  
  @functools.cached_property
  def month_column(self):
    return CategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.Categorical,
      name=f"{self.name} (Month)",
      category_order=MONTHS,
      internal=True
    )

  @functools.cached_property
  def day_of_week_column(self):
    return CategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.Categorical,
      name=f"{self.name} (Day of Week)",
      category_order=DAYS_OF_WEEK,
      internal=True
    )
  
  @functools.cached_property
  def hour_column(self):
    return CategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.Categorical,
      name=f"{self.name} (Hour)",
      internal=True,
      alphanumeric_order=True,
      category_order=HOURS,
    )
  
  def get_internal_columns(self):
    return [
      self.year_column,
      self.month_column,
      self.day_of_week_column,
      self.hour_column,
    ]

  def fit(self, df):
    if not pd.api.types.is_datetime64_any_dtype(df[self.name]):
      kwargs = dict()
      if self.datetime_format is not None:
        kwargs["format"] = self.datetime_format
      datetime_column = pd.to_datetime(df[self.name], **kwargs)
      df[self.name] = datetime_column
    else:
      datetime_column = df[self.name]

    year_column = pd.Categorical(datetime_column.dt.year)
    year_column = year_column.rename_categories({v: str(v) for v in year_column.categories})
    year_column = year_column.reorder_categories(sorted(year_column.categories), ordered=True)
    df[self.year_column.name] = year_column

    month_column = pd.Categorical(datetime_column.dt.month)
    month_column = month_column.rename_categories({k+1: v for k, v in enumerate(MONTHS)})
    df[self.month_column.name] = month_column

    dayofweek_column = pd.Categorical(datetime_column.dt.dayofweek)
    dayofweek_column = dayofweek_column.rename_categories({k: v for k, v in enumerate(DAYS_OF_WEEK)})
    df[self.day_of_week_column.name] = dayofweek_column

    hour_column = pd.Categorical(datetime_column.dt.hour)
    hour_column = hour_column.rename_categories({k: v for k, v in enumerate(HOURS)})
    df[self.hour_column.name] = hour_column

class BaseMultiCategoricalSchemaColumn(BaseSchemaColumn, pydantic.BaseModel, abc.ABC):
  delimiter: str = ","
  is_json: bool = True

  def _convert_string_to_tags_list(self, data: Sequence[str])->list[list[str]]:
    import orjson

    rows: list[list[str]] = []
    for row in data:
      row_categories: list[str]
      if self.is_json:
        row_categories = list(orjson.loads(row))
      else:
        row_categories = list(map(
          lambda category: category.strip(),
          str(row).split(self.delimiter)
        ))
      rows.append(row_categories)
    return rows
  
  def _convert_tags_list_to_json(self, tags_list: Sequence[Sequence[str]]):
    import orjson
    return list(map(
      lambda tags: orjson.dumps(tags),
      tags_list
    ))

class MultiCategoricalSchemaColumn(BaseMultiCategoricalSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.MultiCategorical]
  delimiter: str = ","
  is_json: bool = True

  def fit(self, df):
    data: Sequence[str] = df[self.name].astype(str) # type: ignore
    tags_list = self._convert_string_to_tags_list(data)
    json_strings = self._convert_tags_list_to_json(tags_list)
    df[self.name] = pd.Series(json_strings)
  
class GeospatialSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Geospatial]
  role: GeospatialRoleEnum

  def fit(self, df):
    data = df[self.name].astype(np.float64)
    df[self.name] = data

class UniqueSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Unique]

  def fit(self, df):
    df[self.name] = df[self.name].astype(str)
  
class ImageSchemaColumn(BaseMultiCategoricalSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Image]
  
  def fit(self, df):
    data: Sequence[str] = df[self.name].astype(str) # type: ignore
    tags_list = self._convert_string_to_tags_list(data)
    json_strings = self._convert_tags_list_to_json(tags_list)
    df[self.name] = pd.Series(json_strings)

  
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
    TopicSchemaColumn,
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