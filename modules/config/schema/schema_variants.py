import functools
import math
from typing import Annotated, ClassVar, Literal, Optional, Sequence, Union

import numpy as np
import pydantic
import pandas as pd

from modules.validation import DiscriminatedUnionValidator

from .base import _BaseMultiCategoricalSchemaColumn, _BaseSchemaColumn, GeospatialRoleEnum, SchemaColumnTypeEnum
from .textual import TextPreprocessingConfig, TopicModelingConfig


class ContinuousSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel):
  type: Literal[SchemaColumnTypeEnum.Continuous]
  # Note: bins contain the bin edges. So this should have length bin_count + 1
  bins: Optional[list[float]] = None
  bin_count: int = pydantic.Field(default=3, ge=2)

  @pydantic.field_validator("bins")
  def __validate_bins(self):
    if self.bins is not None:
      return sorted(self.bins)
    return None

  @functools.cached_property
  def bins_column(self):
    return OrderedCategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.OrderedCategorical,
      name=f"{self.name} (Bins)",
      internal=True,
    )
  
  def get_internal_columns(self):
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
      bin_name = str(category + 1).rjust(digit_length, '0')
      bin_categories[category] = f"Bin {bin_name}: {hist_range}"
    
    # Handle the first bin and last bin cases.
    first_histogram_edge_str = self.__format_3f(histogram_edges[0])
    first_bin_id = str(1).rjust(digit_length, '0') 
    bin_categories[0] = f"Bin {first_bin_id}: (-inf, {first_histogram_edge_str})"

    last_histogram_edge_str = self.__format_3f(histogram_edges[len(histogram_edges) - 1])
    last_bin_id = str(len(histogram_edges) + 1).rjust(digit_length, '0')
    bin_categories[len(histogram_edges)] = f"Bin {last_bin_id}: [{last_histogram_edge_str}, inf)"
    
    # Rename the categorical values
    categorical_bins = categorical_bins.rename_categories(bin_categories)
    categorical_bins = categorical_bins.set_categories(sorted(bin_categories.values()), ordered=True)

    return categorical_bins

  def fit(self, df):
    data = df[self.name].astype(np.float64)
    df[self.name] = data
    df[self.bins_column.name] = self.__histogram(data)
  
class CategoricalSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel):
  type: Literal[SchemaColumnTypeEnum.Categorical]
  def fit(self, df):
    raw_data = df[self.name]
    if raw_data.dtype == 'category':
      data = pd.Categorical(raw_data)
    else:  
      data = pd.Categorical(raw_data.astype(str))
    df[self.name] = data
  
class OrderedCategoricalSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel):
  type: Literal[SchemaColumnTypeEnum.OrderedCategorical]
  category_order: Optional[list[str]] = None

  def fit(self, df):
    raw_data = df[self.name]
    if raw_data.dtype == 'category':
      data = pd.Categorical(raw_data)
    else:  
      data = pd.Categorical(raw_data.astype(str))

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
class TopicSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel):
  type: Literal[SchemaColumnTypeEnum.Topic]
  def fit(self, df):
    df[self.name] = df[self.name].astype(np.int32)

class TextualSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel):
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


class TemporalSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel):
  type: Literal[SchemaColumnTypeEnum.Temporal]
  datetime_format: Optional[str]

  MONTHS: ClassVar[list[str]] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
  DAYS_OF_WEEK: ClassVar[list[str]] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  HOURS: ClassVar[list[str]] = list(map(lambda x: f"{str(x).rjust(2, '0')}:00", range(0, 24)))

  @functools.cached_property
  def year_column(self):
    return OrderedCategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.OrderedCategorical,
      name=f"{self.name} (Year)",
      internal=True
    )
  
  @functools.cached_property
  def month_column(self):
    return OrderedCategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.OrderedCategorical,
      name=f"{self.name} (Month)",
      category_order=self.MONTHS,
      internal=True
    )

  @functools.cached_property
  def day_of_week_column(self):
    return OrderedCategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.OrderedCategorical,
      name=f"{self.name} (Day of Week)",
      category_order=self.DAYS_OF_WEEK,
      internal=True
    )
  
  @functools.cached_property
  def hour_column(self):
    return OrderedCategoricalSchemaColumn(
      type=SchemaColumnTypeEnum.OrderedCategorical,
      name=f"{self.name} (Hour)",
      internal=True,
      category_order=self.HOURS,
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
    month_column = month_column.rename_categories({k+1: v for k, v in enumerate(self.MONTHS)})
    df[self.month_column.name] = month_column

    dayofweek_column = pd.Categorical(datetime_column.dt.dayofweek)
    dayofweek_column = dayofweek_column.rename_categories({k: v for k, v in enumerate(self.DAYS_OF_WEEK)})
    df[self.day_of_week_column.name] = dayofweek_column

    hour_column = pd.Categorical(datetime_column.dt.hour)
    hour_column = hour_column.rename_categories({k: v for k, v in enumerate(self.HOURS)})
    df[self.hour_column.name] = hour_column

class MultiCategoricalSchemaColumn(_BaseMultiCategoricalSchemaColumn, pydantic.BaseModel):
  type: Literal[SchemaColumnTypeEnum.MultiCategorical]
  delimiter: str = ","
  is_json: bool = True

  def fit(self, df):
    data: Sequence[str] = df[self.name].astype(str) # type: ignore
    tags_list = self.json2list(data)
    json_strings = self.list2json(tags_list)
    df[self.name] = pd.Series(json_strings)
  
class GeospatialSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel):
  type: Literal[SchemaColumnTypeEnum.Geospatial]
  role: GeospatialRoleEnum

  def fit(self, df):
    data = df[self.name].astype(np.float64)
    df[self.name] = data

class UniqueSchemaColumn(_BaseSchemaColumn, pydantic.BaseModel):
  type: Literal[SchemaColumnTypeEnum.Unique]

  def fit(self, df):
    df[self.name] = df[self.name].astype(str)
  
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
    MultiCategoricalSchemaColumn,
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
  "MultiCategoricalSchemaColumn",
  "ContinuousSchemaColumn",
  "GeospatialSchemaColumn",
]