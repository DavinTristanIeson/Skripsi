from dataclasses import dataclass
import http
from typing import Optional, Sequence, cast
import numpy as np
import pandas as pd
from modules.api import ApiResult
from modules.table.filter_variants import TableFilter
from routes.table.model import (
  DescriptiveStatisticsResource, GetTableColumnAggregateValuesSchema, GetTableGeographicalColumnSchema, GetTableColumnSchema, TableColumnAggregateMethodEnum, TableColumnAggregateValuesResource,
  TableColumnCountsResource, TableColumnFrequencyDistributionResource,
  TableColumnGeographicalPointsResource, TableColumnValuesResource, TableDescriptiveStatisticsResource,
  TableTopicsResource, TableWordsResource, TableWordItemResource
)
from modules.api.wrapper import ApiError
from modules.config import (
  SchemaColumnTypeEnum,
  GeospatialSchemaColumn
)
from modules.project.cache import ProjectCache
from modules.config.schema.base import CATEGORICAL_SCHEMA_COLUMN_TYPES, GeospatialRoleEnum
from modules.config.schema.schema_variants import SchemaColumn, TextualSchemaColumn
from modules.table import TableEngine
from modules.table.filter import TableSort
from modules.topic.model import Topic


@dataclass
class _TableFilterPreprocessResult:
  data: pd.Series
  df: pd.DataFrame
  column: SchemaColumn

@dataclass
class _TableFilterPreprocessModule:
  cache: ProjectCache
  filter: Optional[TableFilter]
  def apply(self, *, column_name: str, supported_types: Optional[list[SchemaColumnTypeEnum]] = None,
  exclude_invalid: bool = True, transform_topics: bool = True)->_TableFilterPreprocessResult:
    config = self.cache.config
    if supported_types is not None:
      column = config.data_schema.assert_of_type(column_name, supported_types)
    else:
      column = config.data_schema.assert_exists(column_name)
    engine = TableEngine(config=config)

    # Sort the results first for ordered categorical and temporal
    sort: Optional[TableSort] = None
    if sort is None and column.is_ordered:
      sort = TableSort(name=column.name, asc=True)
    df = engine.process_workspace(self.filter, sort)

    if column.name not in df.columns:
      raise ApiError(f"The column \"{column.name}\" does not exist in the dataset. There may have been some sort of data corruption in the application.", http.HTTPStatus.NOT_FOUND)
    data = df[column.name]

    if exclude_invalid:
      mask = data.notna()
      if column.type == SchemaColumnTypeEnum.Topic:
        mask = mask & (data != -1)
      data = data[mask]
      df = df[mask]
    
    if len(df) == 0:
      raise ApiError("There are no rows that can be visualized. Perhaps the filter is too strict; try adjusting the filter to be more lax.", http.HTTPStatus.BAD_REQUEST)

    # Use categorical dtype for topic
    if column.type == SchemaColumnTypeEnum.Topic and transform_topics:
      tm_result = self.cache.load_topic(cast(str, column.source_name))
      categorical_data = pd.Categorical(data)
      categorical_data = categorical_data.rename_categories(tm_result.renamer)
      data = pd.Series(categorical_data, name=column.name)

    return _TableFilterPreprocessResult(
      column=column,
      data=data,
      df=df,
    )

def get_column_values(params: GetTableColumnSchema, cache: ProjectCache):
  result = _TableFilterPreprocessModule(
    cache=cache,
    filter=params.filter
  ).apply(
    column_name=params.column
  )

  return ApiResult(
    data=TableColumnValuesResource(
      column=result.column,
      values=result.data.to_list()
    ),
    message=None
  )

def get_column_unique_values(params: GetTableColumnSchema, cache: ProjectCache):
  result = _TableFilterPreprocessModule(
    cache=cache,
    filter=params.filter
  ).apply(
    column_name=params.column
  )

  unique_values = result.data.unique().tolist()

  return ApiResult(
    data=TableColumnValuesResource(
      column=result.column,
      values=unique_values
    ),
    message=None
  )

def get_column_frequency_distribution(params: GetTableColumnSchema, cache: ProjectCache):
  result = _TableFilterPreprocessModule(
    cache=cache,
    filter=params.filter
  ).apply(
    column_name=params.column,
    supported_types=CATEGORICAL_SCHEMA_COLUMN_TYPES
  )

  column = result.column
  data = result.data

  freqdist = data.value_counts()
  # Ensure every value has a stable sort
  freqdist = freqdist.sort_index()

  return ApiResult(
    data=TableColumnFrequencyDistributionResource(
      column=column,
      frequencies=freqdist.values.tolist(),
      categories=list(map(str, freqdist.index.tolist()))
    ),
    message=None
  )

def get_column_aggregate_values(params: GetTableColumnAggregateValuesSchema, cache: ProjectCache):
  config = cache.config
  config.data_schema.assert_of_type(params.column, [
    SchemaColumnTypeEnum.Continuous
  ])

  # We get the grouper instead since there are multiple behaviors for ordered categorical, topic, etc.
  preprocess = _TableFilterPreprocessModule(
    cache=cache,
    filter=params.filter
  ).apply(
    column_name=params.grouped_by,
    supported_types=CATEGORICAL_SCHEMA_COLUMN_TYPES
  )

  df = preprocess.df
  grouper = preprocess.data
  column = preprocess.column

  continuous_data = df[params.column]
  data = pd.concat([grouper, continuous_data], axis=1)
  data.dropna()

  grouped = data.groupby(by=params.grouped_by, sort=True, dropna=True)
  
  if params.method == TableColumnAggregateMethodEnum.Sum:
    data = grouped.sum()
  elif params.method == TableColumnAggregateMethodEnum.StandardDeviation:
    data = grouped.std()
  elif params.method == TableColumnAggregateMethodEnum.Median:
    data = grouped.median()
  elif params.method == TableColumnAggregateMethodEnum.Max:
    data = grouped.max()
  elif params.method == TableColumnAggregateMethodEnum.Min:
    data = grouped.min()
  elif params.method == TableColumnAggregateMethodEnum.Mean:
    data = grouped.mean()
  else:
    raise ValueError(f"\"{params.method}\" is not a valid aggregation method.")
  
  data.dropna(inplace=True)
  if len(data) == 0:
    raise ApiError(f"Oops, there are no valid values in \"{params.grouped_by}\" to group \"{params.column}\".", http.HTTPStatus.BAD_REQUEST)

  return ApiResult(
    data=TableColumnAggregateValuesResource(
      column=column,
      values=data[params.column].tolist(),
      categories=list(map(str, data.index.tolist())),
    ),
    message=None
  )

def get_column_geographical_points(params: GetTableGeographicalColumnSchema, cache: ProjectCache):
  df = cache.load_workspace()
  config = cache.config
  latitude_column = cast(GeospatialSchemaColumn, config.data_schema.assert_of_type(params.latitude_column, [SchemaColumnTypeEnum.Geospatial]))
  longitude_column = cast(GeospatialSchemaColumn, config.data_schema.assert_of_type(params.longitude_column, [SchemaColumnTypeEnum.Geospatial]))
  label_column = config.data_schema.assert_exists(params.label_column) if params.label_column is not None else None

  if latitude_column.role != GeospatialRoleEnum.Latitude:
    raise ApiError(f"\"{latitude_column}\" is a column of type \"Geospatial\", but it does not contain latitude values. Perhaps you meant to use this column as a longitude column?", http.HTTPStatus.UNPROCESSABLE_ENTITY)
  if longitude_column.role != GeospatialRoleEnum.Longitude:
    raise ApiError(f"\"{latitude_column}\" is a column of type \"Geospatial\", but it does not contain longitude values. Perhaps you meant to use this column as a latitude column?", http.HTTPStatus.UNPROCESSABLE_ENTITY)

  engine = TableEngine(config=config)
  filtered_df = engine.filter(df, params.filter)

  check_these_columns = [latitude_column.name, longitude_column.name]
  if label_column is not None:
    check_these_columns.append(label_column.name)
  for column in check_these_columns:
    if column not in filtered_df.columns:
      raise ApiError(f"The column \"{column}\" does not exist in the dataset. There may have been some sort of data corruption in the application.", http.HTTPStatus.NOT_FOUND)

  # Remove NA and invalid values
  latitude_raw = filtered_df[latitude_column.name]
  latitude_mask = latitude_raw.notna()
  longitude_raw = filtered_df[longitude_column.name]
  longitude_mask = longitude_raw.notna()

  coordinate_mask = latitude_mask & longitude_mask

  latitude_masked = latitude_raw[coordinate_mask]
  longitude_masked = longitude_raw[coordinate_mask]
  labels_masked = filtered_df[label_column.name] if label_column is not None else None

  # Count duplicates
  # https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe
  coordinates_raw = [latitude_masked, longitude_masked]
  if labels_masked is not None:
    coordinates_raw.append(labels_masked)
  coordinates = pd.concat(coordinates_raw, axis=1)

  aggregator = dict(size=pd.NamedAgg(column="latitude", aggfunc="size"))
  if label_column is not None:
    aggregator[label_column.name] = pd.NamedAgg(column=label_column.name, aggfunc="first")
  unique_coordinates = (coordinates
    .groupby([latitude_masked.name, longitude_masked.name], as_index=False)
    .agg(**aggregator)) # type: ignore
  
  latitude = unique_coordinates.loc[:, latitude_column.name].to_list()
  longitude = unique_coordinates.loc[:, longitude_column.name].to_list()
  sizes = unique_coordinates.loc[:, "size"].to_list()
  labels: Optional[list[str]] = None
  if label_column is not None:
    labels = list(map(
      lambda label: str(label) if not pd.isna(label) else "Unnamed Location",
      unique_coordinates.loc[:, label_column.name].to_list())
    )

  return ApiResult(
    data=TableColumnGeographicalPointsResource(
      latitude_column=latitude_column,
      longitude_column=longitude_column,
      latitudes=latitude,
      longitudes=longitude,
      labels=labels,
      sizes=sizes,
    ),
    message=None
  )

def get_column_counts(params: GetTableColumnSchema, cache: ProjectCache):
  result = _TableFilterPreprocessModule(
    cache=cache,
    filter=params.filter
  ).apply(
    column_name=params.column,
    exclude_invalid=False,
  )

  data = result.data
  column = result.column
  
  full_df = cache.load_workspace()
  total_count = len(full_df)

  inside_count = len(data)
  notna_count = data.count()
  na_count = inside_count - notna_count

  outlier_count: int | None = None
  if column.type == SchemaColumnTypeEnum.Topic:
    outlier_count = (data == -1).sum()
    notna_count -= outlier_count

  return ApiResult(
    data=TableColumnCountsResource(
      column=column,
      inside=inside_count,
      outside=total_count - inside_count,
      invalid=na_count,
      valid=notna_count,
      total=total_count,
      outlier=outlier_count
    ),
    message=None
  )

def get_column_word_frequencies(params: GetTableColumnSchema, cache: ProjectCache):
  config = cache.config
  column = cast(TextualSchemaColumn, config.data_schema.assert_of_type(params.column, [SchemaColumnTypeEnum.Textual]))

  from modules.topic.bertopic_ext import BERTopicInterpreter
  result = _TableFilterPreprocessModule(
    filter=params.filter,
    cache=cache,
  ).apply(
    column_name=column.preprocess_column.name,
    supported_types=[SchemaColumnTypeEnum.Unique]
  )

  bertopic_model = cache.load_bertopic(column.name)
  interpreter = BERTopicInterpreter(bertopic_model)
  interpreter.top_n_words = 50

  bow = interpreter.represent_as_bow(cast(Sequence[str], result.data))
  # Intentionally only using the BOW rather than C-TF-IDF version
  highest_word_frequencies = interpreter.get_weighted_words(bow)

  word_cloud_items = list(map(lambda word: TableWordItemResource(
    group=0,
    word=word[0],
    size=int(word[1]),
  ), highest_word_frequencies))
  
  return ApiResult(data=TableWordsResource(
    column=column,
    words=word_cloud_items,
  ), message=None)

def get_column_topic_words(params: GetTableColumnSchema, cache: ProjectCache):
  config = cache.config
  column = cast(TextualSchemaColumn, config.data_schema.assert_of_type(params.column, [SchemaColumnTypeEnum.Textual]))

  from modules.topic.bertopic_ext import BERTopicInterpreter
  result = _TableFilterPreprocessModule(
    filter=params.filter,
    cache=cache,
  ).apply(
    column_name=column.topic_column.name,
    supported_types=[SchemaColumnTypeEnum.Topic],
    transform_topics=False
  )
  topics = result.data
  documents = result.df[column.preprocess_column.name]

  bertopic_model = cache.load_bertopic(column.name)
  interpreter = BERTopicInterpreter(bertopic_model)

  bow = interpreter.represent_as_bow_sparse(cast(Sequence[str], documents))
  ctfidf = interpreter.represent_as_ctfidf(bow)
  topic_ctfidfs, unique_topics = interpreter.topic_ctfidfs_per_class(ctfidf, topics)

  tuned_topics: list[Topic] = []
  for topic_id in unique_topics:
    # We're gonna perform argsort, which sparse matrices don't really support. This has to be transformed to np.array
    ctfidf_nparray_row = topic_ctfidfs.getrow(topic_id).toarray()[0]
    words = interpreter.get_weighted_words(ctfidf_nparray_row)
    label = interpreter.get_label(words)
    tuned_topics.append(Topic(
      id=topic_id,
      frequency=(topics == topic_id).sum(),
      label=label or f"Topic {len(tuned_topics) + 1}",
      words=words
    ))

  return ApiResult(data=TableTopicsResource(
    column=column,
    topics=tuned_topics,
  ), message=None)

def get_column_descriptive_statistics(params: GetTableColumnSchema, cache: ProjectCache):
  result = _TableFilterPreprocessModule(
    filter=params.filter,
    cache=cache,
  ).apply(
    column_name=params.column,
    supported_types=[SchemaColumnTypeEnum.Continuous]
  )
  return ApiResult(data=TableDescriptiveStatisticsResource(
    column=result.column,
    statistics=DescriptiveStatisticsResource.from_series(result.data),
  ), message=None)

__all__ = [
  "get_column_frequency_distribution",
  "get_column_geographical_points",
  "get_column_counts",
  "get_column_topic_words",
  "get_column_word_frequencies",
  "get_column_descriptive_statistics",
  "get_column_aggregate_values",
  "get_column_unique_values",
  "get_column_values",
]