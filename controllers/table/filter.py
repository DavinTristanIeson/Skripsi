from typing import cast
import pandas as pd
from modules.api import ApiResult
from models.table import (
  GetTableGeographicalColumnSchema, GetTableColumnSchema,
  TableColumnCountsResource, TableColumnFrequencyDistributionResource,
  TableColumnGeographicalPointsResource, TableColumnValuesResource
)
from modules.config import (
  ProjectCache, SchemaColumnTypeEnum,
  MultiCategoricalSchemaColumn
)
from modules.table import TableEngine, PaginatedApiResult, PaginationParams

def paginate_table(params: PaginationParams, cache: ProjectCache)->PaginatedApiResult:
  df = cache.load_workspace()
  engine = TableEngine(config=cache.config)
  data = engine.paginate(df, params)
  return PaginatedApiResult(
    data=data.to_dict("records"),
    message=None,
    meta=engine.get_meta(data, params)
  )

def get_column_values(params: GetTableColumnSchema, cache: ProjectCache):
  config = cache.config
  df = cache.load_workspace()
  column = config.data_schema.assert_exists(params.column)

  engine = TableEngine(config=config)
  filtered_df = engine.filter(df, params.filter)
  data = filtered_df[params.column]
  data.dropna(inplace=True)

  return ApiResult(
    data=TableColumnValuesResource(
      column=column,
      values=data.to_list()
    ),
    message=None
  )

def get_column_unique_values(params: GetTableColumnSchema, cache: ProjectCache):
  config = cache.config
  df = cache.load_workspace()
  column = config.data_schema.assert_exists(params.column)

  engine = TableEngine(config=config)
  filtered_df = engine.filter(df, params.filter)
  data = filtered_df[params.column]
  data.dropna(inplace=True)

  unique_values = data.unique()

  return ApiResult(
    data=TableColumnValuesResource(
      column=column,
      values=unique_values.tolist()
    ),
    message=None
  )


def get_column_frequency_distribution(params: GetTableColumnSchema, cache: ProjectCache):
  df = cache.load_workspace()
  config = cache.config
  column = config.data_schema.assert_of_type(params.column, [
    SchemaColumnTypeEnum.Categorical,
    SchemaColumnTypeEnum.OrderedCategorical,
    SchemaColumnTypeEnum.Topic,
    SchemaColumnTypeEnum.MultiCategorical,
  ])

  engine = TableEngine(config=config)
  filtered_df = engine.filter(df, params.filter)
  data = filtered_df[column.name]
  data.dropna(inplace=True)

  if column.type == SchemaColumnTypeEnum.MultiCategorical:
    _column = cast(MultiCategoricalSchemaColumn, column)
    freqdist = pd.Series(_column.count_categories(data))
  else:
    freqdist = data.value_counts()

  return ApiResult(
    data=TableColumnFrequencyDistributionResource(
      column=column,
      frequencies=freqdist.values.tolist(),
      values=freqdist.index.tolist()
    ),
    message=None
  )

def get_column_geographical_points(params: GetTableGeographicalColumnSchema, cache: ProjectCache):
  df = cache.load_workspace()
  config = cache.config
  latitude_column = config.data_schema.assert_exists(params.latitude)
  longitude_column = config.data_schema.assert_exists(params.longitude)

  engine = TableEngine(config=config)
  filtered_df = engine.filter(df, params.filter)

  # Remove NA values
  latitude_raw = filtered_df[latitude_column]
  latitude_raw_mask = latitude_raw.notna()
  longitude_raw = filtered_df[longitude_column]
  longitude_raw_mask = longitude_raw.notna()

  coordinate_mask = latitude_raw_mask & longitude_raw_mask
  latitude_raw = latitude_raw[coordinate_mask]
  longitude_raw = longitude_raw[coordinate_mask]

  # Count duplicates
  # https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe
  coordinates = pd.concat([latitude_raw, longitude_raw], axis=1)
  unique_coordinates = coordinates.groupby(coordinates.columns.tolist(), as_index=False).size()
  
  latitude = unique_coordinates.iloc[:, 0].to_list()
  longitude = unique_coordinates.iloc[:, 1].to_list()
  sizes = unique_coordinates.iloc[:, 2].to_list()

  return ApiResult(
    data=TableColumnGeographicalPointsResource(
      latitude_column=latitude_column,
      longitude_column=longitude_column,
      latitude=latitude,
      longitude=longitude,
      sizes=sizes,
    ),
    message=None
  )

def get_column_counts(params: GetTableColumnSchema, cache: ProjectCache):
  df = cache.load_workspace()
  config = cache.config
  column = config.data_schema.assert_exists(params.column)

  engine = TableEngine(config=config)
  filtered_df = engine.filter(df, params.filter)
  data = filtered_df[params.column]

  total_count = len(data)
  notna_count = data.count()
  na_count = total_count - notna_count

  outlier_count: int | None = None
  if column.type == SchemaColumnTypeEnum.Topic:
    outlier_count = (data == -1).sum()
    notna_count -= outlier_count


  return ApiResult(
    data=TableColumnCountsResource(
      column=column,
      invalid=na_count,
      valid=notna_count,
      total=total_count,
      outlier=outlier_count
    ),
    message=None
  )

__all__ = [
  "paginate_table",
  "get_column_values",
  "get_column_frequency_distribution",
  "get_column_geographical_points",
  "get_column_counts",
  "get_column_unique_values",
]