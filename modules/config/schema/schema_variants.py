from enum import Enum
import functools
import math
from typing import Annotated, ClassVar, Literal, Optional, Union

import numpy as np
import pydantic
import pandas as pd

from modules.api.enum import ExposedEnum
from modules.config.context import ConfigSerializationContext
from modules.validation import DiscriminatedUnionValidator

from .base import _BaseSchemaColumn, GeospatialRoleEnum, SchemaColumnTypeEnum
from .textual import TextPreprocessingConfig, TopicModelingConfig


class ContinuousSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel, frozen=True):
  type: Literal[SchemaColumnTypeEnum.Continuous]
  # Note: bins contain the bin edges. So this should have length bin_count + 1
  bins: Optional[list[float]] = None
  bin_count: int = pydantic.Field(default=3, ge=2)

  @pydantic.field_validator("bins", mode="after")
  def __validate_bins(cls, value: Optional[list[float]]):
    if value is not None:
      return sorted(value)
    return None

  @functools.cached_property
  def bins_column(self):
    return OrderedCategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.OrderedCategorical,
      name=f"{self.name} (Bins)",
      internal=True,
      source_name=self.name,
    )
  
  def get_internal_columns(self)->list["SchemaColumn"]:
    return [self.bins_column]
  
  def __format_3f(self, value: float)->str:
    return f"{value:.3f}".rstrip('0').rstrip('.')
  
  def __histogram(self, data: pd.Series):
    # Get histogram edges
    if self.bins is not None:
      histogram_edges = np.array(self.bins)
      histogram_edges.sort()
    else:
      histogram_edges = np.histogram_bin_edges(data, self.bin_count)

    # Split data to bins
    bins = np.digitize(data, histogram_edges)
    categorical_bins = pd.Categorical(bins, categories=np.arange(1, len(histogram_edges)+1), ordered=True)

    # Name the bins so that it can be sorted alphanumerically.
    digit_length = math.floor(math.log10(len(histogram_edges))) + 1

    bin_categories = dict()
    for category in range(1, len(histogram_edges)):
      # Get the edges that define the bin
      current_histogram_edge_str = self.__format_3f(histogram_edges[category]) # Right edge
      prev_histogram_edge_str = self.__format_3f(histogram_edges[category - 1]) # Left edge 
      hist_range = f"[{prev_histogram_edge_str}, {current_histogram_edge_str})" # Range

      # Give the bin an alphanumerically sortable prefix.
      bin_name = str(category).rjust(digit_length, '0')
      bin_categories[category] = f"Bin {bin_name}: {hist_range}"
    
    # Handle the first bin and last bin cases.
    first_histogram_edge_str = self.__format_3f(histogram_edges[0])
    first_bin_id = str(0).rjust(digit_length, '0') 
    bin_categories[0] = f"Bin {first_bin_id}: (-inf, {first_histogram_edge_str})"

    last_histogram_edge_str = self.__format_3f(histogram_edges[len(histogram_edges) - 1])
    last_bin_id = str(len(histogram_edges)).rjust(digit_length, '0')
    bin_categories[len(histogram_edges)] = f"Bin {last_bin_id}: [{last_histogram_edge_str}, inf)"
    
    # Rename the categorical values
    categorical_bins = categorical_bins.rename_categories(bin_categories)
    categorical_bins = categorical_bins.set_categories(sorted(bin_categories.values()), ordered=True)

    return categorical_bins

  def fit(self, df):
    data = df[self.name].astype(pd.Float64Dtype())
    df[self.name] = data
    df[self.bins_column.name] = self.__histogram(data)
  
class CategoricalSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel, frozen=True):
  type: Literal[SchemaColumnTypeEnum.Categorical]
  def fit(self, df):
    raw_data = df[self.name]
    if raw_data.dtype == 'category':
      data = pd.Categorical(raw_data)
    else:  
      data = pd.Categorical(raw_data.astype(pd.StringDtype()))
    df[self.name] = data
  
class OrderedCategoricalSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel, frozen=True):
  type: Literal[SchemaColumnTypeEnum.OrderedCategorical]
  category_order: Optional[list[str]] = None

  def fit(self, df):
    raw_data = df[self.name]
    if raw_data.dtype == 'category':
      data = pd.Categorical(raw_data)
    else:  
      data = pd.Categorical(raw_data.astype(pd.StringDtype()))

    category_order: Optional[list[str]] = None  
    if self.category_order is not None:
      category_order = self.category_order
    else:
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

# Topic column are special because they contain the topic IDs rather than the topic names.
# Topic ID will be mapped by FE with the topic information.
# This makes relabeling or recalculating topic representation much easier.
class TopicSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel, frozen=True):
  type: Literal[SchemaColumnTypeEnum.Topic]
  def fit(self, df):
    df[self.name] = df[self.name].astype(pd.Int32Dtype())

class TextualSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel, frozen=True):
  type: Literal[SchemaColumnTypeEnum.Textual]
  preprocessing: TextPreprocessingConfig
  topic_modeling: TopicModelingConfig

  @functools.cached_property
  def preprocess_column(self):
    return UniqueSchemaColumn(
      type=SchemaColumnTypeEnum.Unique,
      name=f"{self.name} (Preprocessed)",
      internal=True,
      source_name=self.name,
    )

  @functools.cached_property
  def topic_column(self):
    return TopicSchemaColumn(
      type=SchemaColumnTypeEnum.Topic,
      name=f"{self.name} (Topic)",
      internal=True,
      source_name=self.name,
    )
  
  def get_internal_columns(self)->list["SchemaColumn"]:
    return [
      self.preprocess_column,
      self.topic_column,
    ]
  
  def fit(self, df):
    data = df[self.name].astype(pd.StringDtype())
    df[self.name] = data


class TemporalColumnFeatureEnum(str, Enum):
  # Non-repeating
  Year = "year"
  Month = "month"
  Date = "date"

  Monthly = "monthly"
  DayOfWeek = "day_of_week"
  Hour = "hour"

# Only used by FE for formatting purposes
class TemporalPrecisionEnum(str, Enum):
  Year = "year"
  Month = "month"
  Date = "date"

ExposedEnum().register(TemporalColumnFeatureEnum)
ExposedEnum().register(TemporalPrecisionEnum)

class TemporalSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel, frozen=True):
  type: Literal[SchemaColumnTypeEnum.Temporal]
  datetime_format: Optional[str]
  temporal_features: list[TemporalColumnFeatureEnum]
  temporal_precision: Optional[TemporalPrecisionEnum] = None

  @pydantic.field_serializer("temporal_precision")
  def __serialize_precision(value, info: pydantic.SerializationInfo):
    if isinstance(info.context, ConfigSerializationContext) and info.context.is_save:
      return None
    return value

  MONTHS: ClassVar[list[str]] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
  DAYS_OF_WEEK: ClassVar[list[str]] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
  HOURS: ClassVar[list[str]] = list(map(lambda x: f"{str(x).rjust(2, '0')}:00", range(0, 24)))

  @functools.cached_property
  def year_column(self):
    return TemporalSchemaColumn(
      type=SchemaColumnTypeEnum.Temporal,
      temporal_features=[],
      temporal_precision=TemporalPrecisionEnum.Year,
      datetime_format=None,
      name=f"{self.name} (Year)",
      internal=True,
      source_name=self.name,
    )
  
  @functools.cached_property
  def month_column(self):
    return TemporalSchemaColumn(
      type=SchemaColumnTypeEnum.Temporal,
      temporal_precision=TemporalPrecisionEnum.Month,
      temporal_features=[],
      datetime_format=None,
      name=f"{self.name} (Month)",
      internal=True,
      source_name=self.name,
    )
  
  @functools.cached_property
  def date_column(self):
    return TemporalSchemaColumn(
      type=SchemaColumnTypeEnum.Temporal,
      temporal_precision=TemporalPrecisionEnum.Date,
      temporal_features=[],
      datetime_format=None,
      name=f"{self.name} (Date)",
      internal=True,
      source_name=self.name,
    )
  
  @functools.cached_property
  def monthly_column(self):
    return OrderedCategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.OrderedCategorical,
      name=f"{self.name} (Monthly)",
      category_order=self.MONTHS,
      internal=True,
      source_name=self.name,
    )

  @functools.cached_property
  def day_of_week_column(self):
    return OrderedCategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.OrderedCategorical,
      name=f"{self.name} (Day of Week)",
      category_order=self.DAYS_OF_WEEK,
      internal=True,
      source_name=self.name,
    )
  
  @functools.cached_property
  def hour_column(self):
    return OrderedCategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.OrderedCategorical,
      name=f"{self.name} (Hour)",
      internal=True,
      category_order=self.HOURS,
      source_name=self.name,
    )
  
  def get_internal_columns(self)->list["SchemaColumn"]:
    if self.internal:
      return []
    
    internal_columns = []
    if TemporalColumnFeatureEnum.Year in self.temporal_features:
      internal_columns.append(self.year_column)
    if TemporalColumnFeatureEnum.Month in self.temporal_features:
      internal_columns.append(self.month_column)
    if TemporalColumnFeatureEnum.Date in self.temporal_features:
      internal_columns.append(self.date_column)
    if TemporalColumnFeatureEnum.Monthly in self.temporal_features:
      internal_columns.append(self.monthly_column)
    if TemporalColumnFeatureEnum.DayOfWeek in self.temporal_features:
      internal_columns.append(self.day_of_week_column)
    if TemporalColumnFeatureEnum.Hour in self.temporal_features:
      internal_columns.append(self.hour_column)
    
    return internal_columns

  def fit(self, df):
    if not pd.api.types.is_datetime64_any_dtype(df[self.name]):
      kwargs = dict()
      if self.datetime_format is not None:
        kwargs["format"] = self.datetime_format
      datetime_column = pd.to_datetime(df[self.name], **kwargs)
    else:
      datetime_column = df[self.name]

    df[self.name] = datetime_column
    if TemporalColumnFeatureEnum.Year in self.temporal_features:
      # https://github.com/pandas-dev/pandas/issues/15303
      # Round doesn't work for Y and M since they're not fixed frequencies.
      year_column = datetime_column.dt.to_period("M").dt.to_timestamp()
      df[self.year_column.name] = year_column
    
    if TemporalColumnFeatureEnum.Month in self.temporal_features:
      month_column = datetime_column.dt.to_period("M").dt.to_timestamp()
      df[self.month_column.name] = month_column
    
    if TemporalColumnFeatureEnum.Date in self.temporal_features:
      date_column = datetime_column.dt.round("D")
      df[self.date_column.name] = date_column
    
    if TemporalColumnFeatureEnum.Monthly in self.temporal_features:
      monthly_column = pd.Categorical(datetime_column.dt.month)
      monthly_column = monthly_column.rename_categories({k+1: v for k, v in enumerate(self.MONTHS)})
      df[self.monthly_column.name] = monthly_column
    
    if TemporalColumnFeatureEnum.DayOfWeek in self.temporal_features:
      dayofweek_column = pd.Categorical(datetime_column.dt.dayofweek)
      dayofweek_column = dayofweek_column.rename_categories({k: v for k, v in enumerate(self.DAYS_OF_WEEK)})
      df[self.day_of_week_column.name] = dayofweek_column
    
    if TemporalColumnFeatureEnum.Hour in self.temporal_features:
      hour_column = pd.Categorical(datetime_column.dt.hour)
      hour_column = hour_column.rename_categories({k: v for k, v in enumerate(self.HOURS)})
      df[self.hour_column.name] = hour_column

    print(df[self.year_column.name], df[self.month_column.name])

class GeospatialSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel, frozen=True):
  type: Literal[SchemaColumnTypeEnum.Geospatial]
  role: GeospatialRoleEnum

  def fit(self, df):
    data = df[self.name].astype(pd.Float64Dtype())
    df[self.name] = data

class UniqueSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel, frozen=True):
  type: Literal[SchemaColumnTypeEnum.Unique]

  def fit(self, df):
    df[self.name] = df[self.name].astype(pd.StringDtype())
  
SchemaColumn = Annotated[
  Union[
    UniqueSchemaColumn,
    CategoricalSchemaColumn,
    OrderedCategoricalSchemaColumn,
    TextualSchemaColumn,
    ContinuousSchemaColumn,
    TemporalSchemaColumn,
    GeospatialSchemaColumn,
    UniqueSchemaColumn,
    TopicSchemaColumn,
  ],
  pydantic.Field(discriminator="type"),
  DiscriminatedUnionValidator
]

__all__ = [
  "SchemaColumn",
  "TextualSchemaColumn",
  "UniqueSchemaColumn",
  "OrderedCategoricalSchemaColumn",
  "CategoricalSchemaColumn",
  "ContinuousSchemaColumn",
  "GeospatialSchemaColumn",
]