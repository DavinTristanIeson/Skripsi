from dataclasses import dataclass
import http
from typing import Optional, Sequence, cast
import pandas as pd
from modules.api import ApiResult
from modules.table.filter_variants import TableFilter
from routes.table.model import (
  DescriptiveStatisticsResource, GetTableColumnAggregateTotalsSchema, GetTableGeographicalColumnSchema, GetTableColumnSchema, TableColumnAggregateTotalsResource,
  TableColumnCountsResource, TableColumnFrequencyDistributionResource,
  TableColumnGeographicalPointsResource, TableColumnValuesResource, TableDescriptiveStatisticsResource,
  TableTopicsResource, TableWordsResource, TableWordItemResource
)
from modules.api.wrapper import ApiError
from modules.config import (
  SchemaColumnTypeEnum,
  MultiCategoricalSchemaColumn, GeospatialSchemaColumn
)
from modules.project.cache import ProjectCache, ProjectCacheManager
from modules.config.schema.base import GeospatialRoleEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.table import TableEngine, TablePaginationApiResult, PaginationParams
from modules.table.filter import TableSort
from modules.topic.model import Topic, TopicModelingResult

def _filter_table(
  *,
  column_name: str,
  filter: Optional[TableFilter],
  cache: ProjectCache,
  supported_types: Optional[list[SchemaColumnTypeEnum]] = None,
):
  config = cache.config
  if supported_types is not None:
    column = config.data_schema.assert_of_type(column_name, supported_types)
  else:
    column = config.data_schema.assert_exists(column_name)
  engine = TableEngine(config=config)

  # Sort the results first for ordered categorical and temporal
  sort: Optional[TableSort] = None
  if sort is None and column.type == SchemaColumnTypeEnum.OrderedCategorical or column.type == SchemaColumnTypeEnum.Temporal:
    sort = TableSort(name=column.name, asc=True)
  df = engine.process_workspace(filter, sort)

  if column.name not in df.columns:
    raise ApiError(f"The column \"{column.name}\" does not exist in the dataset. There may have been some sort of data corruption in the application.", http.HTTPStatus.NOT_FOUND)
  data = df[column.name]

  mask = data.notna()
  if column.type == SchemaColumnTypeEnum.Topic:
    mask = data == -1
  data = data[mask]
  df = df[mask]

  # Use categorical dtype for topic
  if column.type == SchemaColumnTypeEnum.Topic:
    tm_result = cache.load_topic(cast(str, column.source_name))
    categorical_data = pd.Categorical(data)
    data = cast(pd.Series, categorical_data.rename_categories(tm_result.renamer))

  return data, df, column


def get_column_values(params: GetTableColumnSchema, cache: ProjectCache):
  data, df, column = _filter_table(
    column_name=params.column,
    filter=params.filter,
    cache=cache
  )

  return ApiResult(
    data=TableColumnValuesResource(
      column=column,
      values=data.to_list()
    ),
    message=None
  )

def get_column_unique_values(params: GetTableColumnSchema, cache: ProjectCache):
  data, df, column = _filter_table(
    column_name=params.column,
    filter=params.filter,
    cache=cache
  )

  unique_values = data.unique().tolist()

  return ApiResult(
    data=TableColumnValuesResource(
      column=column,
      values=unique_values
    ),
    message=None
  )

def get_column_frequency_distribution(params: GetTableColumnSchema, cache: ProjectCache):
  data, df, column = _filter_table(
    column_name=params.column,
    filter=params.filter,
    cache=cache,
    supported_types=[
      SchemaColumnTypeEnum.Categorical,
      SchemaColumnTypeEnum.OrderedCategorical,
      SchemaColumnTypeEnum.Topic,
      SchemaColumnTypeEnum.Temporal,
    ]
  )

  freqdist = data.value_counts()

  return ApiResult(
    data=TableColumnFrequencyDistributionResource(
      column=column,
      frequencies=freqdist.values.tolist(),
      values=freqdist.index.tolist()
    ),
    message=None
  )

def get_column_aggregate_totals(params: GetTableColumnAggregateTotalsSchema, cache: ProjectCache):
  config = cache.config
  config.data_schema.assert_of_type(params.column, [
    SchemaColumnTypeEnum.Continuous
  ])

  # We get the grouper instead since there are multiple behaviors for ordered categorical, topic, etc.
  grouper, df, column = _filter_table(
    column_name=params.grouped_by,
    filter=params.filter,
    cache=cache,
    supported_types=[
      SchemaColumnTypeEnum.Categorical,
      SchemaColumnTypeEnum.OrderedCategorical,
      SchemaColumnTypeEnum.Topic,
      SchemaColumnTypeEnum.Temporal,
    ]
  )

  continuous_data = df[params.column]
  data = pd.concat([grouper, continuous_data], axis=1)
  data.dropna()

  totals = df.groupby(by=params.grouped_by, sort=True, dropna=True).sum()

  return ApiResult(
    data=TableColumnAggregateTotalsResource(
      column=column,
      totals=totals.values.tolist(),
      values=totals.index.tolist(),
    ),
    message=None
  )

def get_column_geographical_points(params: GetTableGeographicalColumnSchema, cache: ProjectCache):
  df = cache.load_workspace()
  config = cache.config
  latitude_column = cast(GeospatialSchemaColumn, config.data_schema.assert_exists(params.latitude_column))
  longitude_column = cast(GeospatialSchemaColumn, config.data_schema.assert_exists(params.longitude_column))
  if latitude_column.role != GeospatialRoleEnum.Latitude:
    raise ApiError(f"\"{latitude_column}\" is a column of type \"Geospatial\", but it does not contain latitude values. Perhaps you meant to use this column as a longitude column?", http.HTTPStatus.UNPROCESSABLE_ENTITY)
  if longitude_column.role != GeospatialRoleEnum.Longitude:
    raise ApiError(f"\"{latitude_column}\" is a column of type \"Geospatial\", but it does not contain longitude values. Perhaps you meant to use this column as a latitude column?", http.HTTPStatus.UNPROCESSABLE_ENTITY)

  engine = TableEngine(config=config)
  filtered_df = engine.filter(df, params.filter)

  # Remove NA and invalid values
  latitude_raw = filtered_df[latitude_column]
  latitude_not_na_mask = latitude_raw.notna()
  latitude_valid_mask = (latitude_raw >= -90) & (latitude_raw <= 90)
  latitude_mask = latitude_not_na_mask & latitude_valid_mask

  longitude_raw = filtered_df[longitude_column]
  longitude_not_na_mask = longitude_raw.notna()
  longitude_valid_mask = (longitude_raw >= -180) & (longitude_raw <= 180)
  longitude_mask = longitude_not_na_mask & longitude_valid_mask

  coordinate_mask = longitude_mask & longitude_mask

  latitude_masked = latitude_raw[coordinate_mask]
  longitude_masked = longitude_raw[coordinate_mask]

  # Count duplicates
  # https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe
  coordinates = pd.concat([latitude_masked, longitude_masked], axis=1)
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
  data, df, column = _filter_table(
    column_name=params.column,
    filter=params.filter,
    cache=cache,
  )

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

def get_column_word_frequencies(params: GetTableColumnSchema, cache: ProjectCache):
  config = cache.config
  column = cast(TextualSchemaColumn, config.data_schema.assert_of_type(params.column, [SchemaColumnTypeEnum.Textual]))

  from modules.topic.bertopic_ext import BERTopicInterpreter
  data, df, column = _filter_table(
    column_name=column.preprocess_column.name,
    filter=params.filter,
    cache=cache,
    supported_types=[SchemaColumnTypeEnum.Unique]
  )

  bertopic_model = cache.load_bertopic(column.name)
  interpreter = BERTopicInterpreter(bertopic_model)
  interpreter.top_n_words = 100

  bow = interpreter.represent_as_bow(cast(Sequence[str], data))
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
  topics, df, _ = _filter_table(
    column_name=column.topic_column.name,
    filter=params.filter,
    cache=cache,
    supported_types=[SchemaColumnTypeEnum.Topic]
  )
  documents = df[column.preprocess_column.name]


  bertopic_model = cache.load_bertopic(column.name)
  interpreter = BERTopicInterpreter(bertopic_model)

  bow = interpreter.represent_as_bow(cast(Sequence[str], documents))
  ctfidf = interpreter.represent_as_ctfidf(bow)
  topic_ctfidfs, unique_topics = interpreter.topic_ctfidfs_per_class(ctfidf, topics)

  tuned_topics: list[Topic] = []
  for topic_id in unique_topics:
    words = interpreter.get_weighted_words(topic_ctfidfs[topic_id])
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
  data, df, column = _filter_table(params, cache, supported_types=[SchemaColumnTypeEnum.Continuous])
  return ApiResult(data=TableDescriptiveStatisticsResource(
    column=column,
    statistics=DescriptiveStatisticsResource.from_series(data),
  ), message=None)

__all__ = [
  "get_column_frequency_distribution",
  "get_column_geographical_points",
  "get_column_counts",
  "get_column_topic_words",
  "get_column_word_frequencies",
  "get_column_descriptive_statistics",
  "get_column_aggregate_totals",
  "get_column_unique_values",
  "get_column_values",
]