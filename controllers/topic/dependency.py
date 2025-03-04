import http
import os
from typing import Annotated

from fastapi import Depends
from controllers.project.dependency import ProjectCacheDependency
from modules.api.wrapper import ApiError
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.topic.model import Topic, TopicModelingResult


def _is_textual_column(column: str, cache: ProjectCacheDependency):
  col = cache.config.data_schema.assert_of_type(column, [SchemaColumnTypeEnum.Textual])
  return col

TextualSchemaColumnDependency = Annotated[TextualSchemaColumn, Depends(_is_textual_column)]

def _has_topic_modeling_result(column: str, cache: ProjectCacheDependency)->TopicModelingResult:
  return cache.load_topic(column)

TopicModelingResultDependency = Annotated[TopicModelingResult, Depends(_has_topic_modeling_result)]

def _topic_exists(topic_modeling_result: TopicModelingResultDependency, topic: int)->list[Topic]:
  base_topics = topic_modeling_result.find(topic)
  if len(base_topics) == 0:
    raise ApiError(f"We cannot find any topic or super-topic with ID: {topic}. Perhaps the file containing the topic modeling result has been corrupted; in which case, please run the topic modeling procedure again. Alternatively, consider refreshing the page to get the newest topic information.", http.HTTPStatus.BAD_REQUEST)
  return base_topics

TopicExistsDependency = Annotated[list[Topic], Depends(_topic_exists)]

__all__ = [
  "TextualSchemaColumnDependency",
  "TopicModelingResultDependency",
  "TopicExistsDependency"
]