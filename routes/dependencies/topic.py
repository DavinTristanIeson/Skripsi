import http
import os
from typing import Annotated, Optional, cast

from fastapi import Depends, Query
import pandas as pd
from routes.dependencies.project import ProjectCacheDependency
from modules.api.wrapper import ApiError
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.topic.model import Topic, TopicModelingResult

def _is_textual_column(column: Annotated[str, Query], cache: ProjectCacheDependency):
  col = cache.config.data_schema.assert_of_type(column, [SchemaColumnTypeEnum.Textual])
  return col

TextualSchemaColumnDependency = Annotated[TextualSchemaColumn, Depends(_is_textual_column)]

def _has_topic_modeling_result(column: str, cache: ProjectCacheDependency)->TopicModelingResult:
  return cache.topics.load(column)

TopicModelingResultDependency = Annotated[TopicModelingResult, Depends(_has_topic_modeling_result)]


def _has_topic_modeling_result_if_topic(column: str, cache: ProjectCacheDependency)->Optional[TopicModelingResult]:
  topic_column = cache.config.data_schema.assert_exists(column)
  if topic_column.type == SchemaColumnTypeEnum.Topic:
    return cache.topics.load(cast(str, topic_column.source_name))
  else:
    return None

OptionalTopicModelingResultDependency = Annotated[TopicModelingResult, Depends(_has_topic_modeling_result)]

def _topic_exists(topic_modeling_result: TopicModelingResultDependency, topic: int = Query())->Topic:
  topic_obj = topic_modeling_result.find(topic)
  if topic_obj is None:
    raise ApiError(f"We cannot find any topic with ID: {topic}. Perhaps the file containing the topic modeling result has been corrupted; in which case, please run the topic modeling procedure again. Alternatively, consider refreshing the page to get the newest topic information.", http.HTTPStatus.BAD_REQUEST)
  return topic_obj

TopicExistsDependency = Annotated[Topic, Depends(_topic_exists)]

__all__ = [
  "TextualSchemaColumnDependency",
  "TopicModelingResultDependency",
  "TopicExistsDependency"
]