import http
from typing import Optional, Sequence, cast
import pandas as pd
from modules.api import ApiResult
from modules.exceptions.dependencies import InvalidValueTypeException
from routes.table.controller.preprocess import TablePreprocessModule, TablePreprocessModule
from routes.table.model import (
  DescriptiveStatisticsResource, GetTableColumnAggregateValuesSchema, GetTableGeographicalAggregateValuesSchema, GetTableGeographicalColumnSchema, GetTableColumnSchema, TableColumnAggregateMethodEnum, TableColumnAggregateValuesResource,
  TableColumnCountsResource, TableColumnFrequencyDistributionResource,
  TableColumnGeographicalPointsResource, TableColumnValuesResource, TableDescriptiveStatisticsResource,
  TableTopicsResource, TableWordFrequenciesResource
)
from modules.api.wrapper import ApiError
from modules.config import (
  SchemaColumnTypeEnum
)
from modules.project.cache import ProjectCache
from modules.config.schema.base import CATEGORICAL_SCHEMA_COLUMN_TYPES
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.topic.model import Topic


def get_column_values(params: GetTableColumnSchema, cache: ProjectCache):
  result = TablePreprocessModule(
    cache=cache,
  ).apply(
    column_name=params.column,
    filter=params.filter
  )

  return ApiResult(
    data=TableColumnValuesResource(
      column=result.column,
      values=result.data.to_list()
    ),
    message=None
  )

def get_column_unique_values(params: GetTableColumnSchema, cache: ProjectCache):
  result = TablePreprocessModule(
    cache=cache,
  ).apply(
    filter=params.filter,
    column_name=params.column,
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
  result = TablePreprocessModule(
    cache=cache,
  ).apply(
    filter=params.filter,
    column_name=params.column,
    supported_types=CATEGORICAL_SCHEMA_COLUMN_TYPES
  )

  column = result.column
  data = result.data

  freqdist = data.value_counts()
  # Ensure every value has a stable sort
  freqdist = freqdist.sort_index(ascending=result.column.type != SchemaColumnTypeEnum.Boolean)

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
    SchemaColumnTypeEnum.Continuous,
  ])

  # We get the grouper instead since there are multiple behaviors for ordered categorical, topic, etc.
  preprocess = TablePreprocessModule(
    cache=cache,
  ).apply(
    filter=params.filter,
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
    raise InvalidValueTypeException(
      value=params.method,
      type="aggregation method",
    )
  
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
  aggregator = dict(size=pd.NamedAgg(column=params.latitude_column, aggfunc="size"))
  column_constraints: dict[str, list[SchemaColumnTypeEnum]] = dict()
  if params.label_column is not None:
    aggregator[params.label_column] = pd.NamedAgg(column=params.label_column, aggfunc="first")
    column_constraints[params.label_column] = [SchemaColumnTypeEnum.Unique, SchemaColumnTypeEnum.Categorical, SchemaColumnTypeEnum.OrderedCategorical]

  preprocess = TablePreprocessModule(
    cache=cache,
  ).apply_geographical(
    filter=params.filter,
    additional_column_constraints=column_constraints,
    aggregator=aggregator,
    latitude_column_name=params.latitude_column,
    longitude_column_name=params.longitude_column,
  )
  
  sizes = preprocess.df.loc[:, "size"].to_list()
  labels: Optional[list[str]] = None
  if params.label_column is not None:
    labels = list(map(
      lambda label: str(label) if not pd.isna(label) else "Unnamed Location",
      preprocess.df.loc[:, params.label_column].to_list())
    )

  return ApiResult(
    data=TableColumnGeographicalPointsResource(
      latitude_column=preprocess.latitude_column,
      longitude_column=preprocess.longitude_column,
      latitudes=preprocess.latitudes,
      longitudes=preprocess.longitudes,
      labels=labels,
      values=sizes,
    ),
    message=None
  )

def get_column_geographical_aggregate_values(params: GetTableGeographicalAggregateValuesSchema, cache: ProjectCache):
  aggregator = {
    params.target_column: pd.NamedAgg(column=params.target_column, aggfunc=params.method),
  }
  column_constraints: dict[str, list[SchemaColumnTypeEnum]] = {
    params.target_column: [SchemaColumnTypeEnum.Continuous]
  }
  if params.label_column is not None:
    aggregator[params.label_column] = pd.NamedAgg(column=params.label_column, aggfunc="first")
    column_constraints[params.label_column] = [SchemaColumnTypeEnum.Unique, SchemaColumnTypeEnum.Categorical, SchemaColumnTypeEnum.OrderedCategorical]

  preprocess = TablePreprocessModule(
    cache=cache,
  ).apply_geographical(
    filter=params.filter,
    additional_column_constraints=column_constraints,
    aggregator=aggregator,
    latitude_column_name=params.latitude_column,
    longitude_column_name=params.longitude_column,
  )

  values = preprocess.df.loc[:, params.target_column].fillna(0).to_list()
  labels: Optional[list[str]] = None
  if params.label_column is not None:
    labels = list(map(
      lambda label: str(label) if not pd.isna(label) else "Unnamed Location",
      preprocess.df.loc[:, params.label_column].to_list())
    )

  return ApiResult(
    data=TableColumnGeographicalPointsResource(
      latitude_column=preprocess.latitude_column,
      longitude_column=preprocess.longitude_column,
      latitudes=preprocess.latitudes,
      longitudes=preprocess.longitudes,
      labels=labels,
      values=values,
    ),
    message=None
  )

def get_column_counts(params: GetTableColumnSchema, cache: ProjectCache):
  result = TablePreprocessModule(
    cache=cache,
  ).apply(
    filter=params.filter,
    column_name=params.column,
    exclude_invalid=False,
  )

  data = result.data
  column = result.column
  
  full_df = cache.workspaces.load()
  total_count = len(full_df)

  inside_count = len(data)
  notna_count = data.count()

  outlier_count: int | None = None
  true_count: int | None = None
  false_count: int | None = None
  if result.column.type == SchemaColumnTypeEnum.Boolean:
    true_count = (data == True).sum()
    false_count = (data == False).sum()
    notna_count -= (true_count + false_count)
  elif column.type == SchemaColumnTypeEnum.Topic:
    outlier_count = (data == -1).sum()
    notna_count -= outlier_count

  na_count = inside_count - notna_count

  return ApiResult(
    data=TableColumnCountsResource(
      column=column,

      total=total_count,
      inside=inside_count,
      outside=total_count - inside_count,

      invalid=na_count,
      valid=notna_count,

      outlier=outlier_count,
      true=true_count,
      false=false_count,
    ),
    message=None
  )

def get_column_word_frequencies(params: GetTableColumnSchema, cache: ProjectCache):
  config = cache.config
  column = cast(TextualSchemaColumn, config.data_schema.assert_of_type(params.column, [SchemaColumnTypeEnum.Textual]))

  from modules.topic.bertopic_ext import BERTopicInterpreter
  result = TablePreprocessModule(
    cache=cache,
  ).apply(
    filter=params.filter,
    column_name=column.preprocess_column.name,
    supported_types=[SchemaColumnTypeEnum.Unique]
  )

  bertopic_model = cache.bertopic_models.load(column.name)
  interpreter = BERTopicInterpreter(bertopic_model)
  interpreter.top_n_words = 50

  bow = interpreter.represent_as_bow(cast(Sequence[str], result.data))
  # Intentionally only using the BOW rather than C-TF-IDF version
  highest_word_frequencies = interpreter.get_weighted_words(bow)

  word_cloud_items = list(map(lambda word: (word[0], int(word[1])), highest_word_frequencies))
  
  return ApiResult(data=TableWordFrequenciesResource(
    column=column,
    words=word_cloud_items,
  ), message=None)

def get_column_topic_words(params: GetTableColumnSchema, cache: ProjectCache):
  config = cache.config
  column = cast(TextualSchemaColumn, config.data_schema.assert_of_type(params.column, [SchemaColumnTypeEnum.Textual]))

  from modules.topic.bertopic_ext import BERTopicInterpreter
  result = TablePreprocessModule(
    cache=cache,
  ).apply(
    filter=params.filter,
    column_name=column.topic_column.name,
    supported_types=[SchemaColumnTypeEnum.Topic],
    transform_data=False
  )
  topics = result.data
  column.assert_internal_columns(result.df, with_preprocess=True, with_topics=False)
  documents = result.df[column.preprocess_column.name]

  bertopic_model = cache.bertopic_models.load(column.name)
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
  result = TablePreprocessModule(
    cache=cache,
  ).apply(
    filter=params.filter,
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