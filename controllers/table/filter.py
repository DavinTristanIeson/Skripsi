import http
from typing import Optional, Sequence, cast
import pandas as pd
from modules.api import ApiResult
from models.table import (
  GetTableGeographicalColumnSchema, GetTableColumnSchema,
  TableColumnCountsResource, TableColumnFrequencyDistributionResource,
  TableColumnGeographicalPointsResource, TableColumnValuesResource,
  TableTopicsResource, TableWordCloudResource, WordCloudItemResource
)
from modules.api.wrapper import ApiError
from modules.config import (
  ProjectCache, SchemaColumnTypeEnum,
  MultiCategoricalSchemaColumn, GeospatialSchemaColumn
)
from modules.config.schema.base import GeospatialRoleEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.table import TableEngine, PaginatedApiResult, PaginationParams
from modules.topic.model import Topic

def paginate_table(params: PaginationParams, cache: ProjectCache)->PaginatedApiResult:
  df = cache.load_workspace()
  engine = TableEngine(cache.config)
  data = engine.paginate(df, params)
  return PaginatedApiResult(
    data=data.to_dict("records"),
    message=None,
    meta=engine.get_meta(data, params)
  )

def _filter_table(params: GetTableColumnSchema, cache: ProjectCache, *, supported_types: Optional[list[SchemaColumnTypeEnum]] = None):
  config = cache.config
  df = cache.load_workspace()
  if supported_types is not None:
    column = config.data_schema.assert_of_type(params.column, supported_types)
  else:
    column = config.data_schema.assert_exists(params.column)

  engine = TableEngine(config=config)
  filtered_df = engine.filter(df, params.filter)
  data = filtered_df[column.name]

  mask = data.notna()
  data = data[mask]
  filtered_df = filtered_df[mask]

  return data, filtered_df, column

def _filter_table_textual(params: GetTableColumnSchema, cache: ProjectCache):
  config = cache.config
  df = cache.load_workspace()
  column = config.data_schema.assert_of_type(params.column, [SchemaColumnTypeEnum.Textual])
  
  engine = TableEngine(config=config)
  filtered_df = engine.filter(df, params.filter)
  column = cast(TextualSchemaColumn, column)
  if column.preprocess_column.name not in filtered_df.categories:
    raise ApiError("The topic modeling procedure has not been executed on this dataset.", http.HTTPStatus.BAD_REQUEST)
  data = filtered_df[column.preprocess_column.name]
  return data, filtered_df, column


def _filter_table_textual_with_topic(params: GetTableColumnSchema, cache: ProjectCache):
  data, df, column = _filter_table_textual(params, cache)
  if column.topic_column.name not in df.categories:
    raise ApiError("The topic modeling procedure has not been executed on this dataset.", http.HTTPStatus.BAD_REQUEST)
  topics = df[column.topic_column.name]
  mask = topics.notna() & topics != -1
  data = data[mask]
  topics = topics[mask]
  df = df[mask]
  return data, topics, df, column

def get_column_values(params: GetTableColumnSchema, cache: ProjectCache):
  data, df, column = _filter_table(params, cache)

  return ApiResult(
    data=TableColumnValuesResource(
      column=column,
      values=data.to_list()
    ),
    message=None
  )

def get_column_unique_values(params: GetTableColumnSchema, cache: ProjectCache):
  data, df, column = _filter_table(params, cache)
  unique_values = data.unique()

  return ApiResult(
    data=TableColumnValuesResource(
      column=column,
      values=unique_values.tolist()
    ),
    message=None
  )


def get_column_frequency_distribution(params: GetTableColumnSchema, cache: ProjectCache):
  data, df, column = _filter_table(params, cache, supported_types=[
    SchemaColumnTypeEnum.Categorical,
    SchemaColumnTypeEnum.OrderedCategorical,
    SchemaColumnTypeEnum.Topic,
    SchemaColumnTypeEnum.MultiCategorical,
  ])
  
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

  coordinate_mask = longitude_not_na_mask & longitude_valid_mask

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
  data, df, column = _filter_table(params, cache)

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
  from modules.topic import BERTopicInterpreter
  data, df, column = _filter_table_textual(params, cache)

  bertopic_model = cache.load_bertopic(column.name)
  interpreter = BERTopicInterpreter(bertopic_model)
  interpreter.top_n_words = 100

  bow = interpreter.represent_as_bow(cast(Sequence[str], data))
  # Intentionally only using the BOW rather than C-TF-IDF version
  highest_word_frequencies = interpreter.get_weighted_words(bow)

  word_cloud_items = list(map(lambda word: WordCloudItemResource(
    color=0,
    word=word[0],
    size=int(word[1]),
  ), highest_word_frequencies))
  
  return TableWordCloudResource(
    column=column,
    words=word_cloud_items,
  )

def get_column_topic_words(params: GetTableColumnSchema, cache: ProjectCache):
  from modules.topic import BERTopicInterpreter
  data, topics, df, column = _filter_table_textual_with_topic(params, cache)

  bertopic_model = cache.load_bertopic(column.name)
  interpreter = BERTopicInterpreter(bertopic_model)

  bow = interpreter.represent_as_bow(cast(Sequence[str], data))
  ctfidf = interpreter.represent_as_ctfidf(bow)
  topic_ctfidfs, unique_topics = interpreter.topic_ctfidfs_per_class(ctfidf, topics)

  tuned_topics: list[Topic] = []
  for topic_id in unique_topics:
    words = interpreter.get_weighted_words(topic_ctfidfs[topic_id])
    tuned_topics.append(Topic(
      id=topic_id,
      frequency=(topics == topic_id).sum(),
      label=interpreter.get_label(topic_ctfidfs) or f'Topic {topic_id}',
      words=words
    ))

  return TableTopicsResource(
    column=column,
    topics=tuned_topics,
  )


__all__ = [
  "paginate_table",
  "get_column_values",
  "get_column_frequency_distribution",
  "get_column_geographical_points",
  "get_column_counts",
  "get_column_unique_values",
  "get_column_topic_words",
  "get_column_word_frequencies",
]