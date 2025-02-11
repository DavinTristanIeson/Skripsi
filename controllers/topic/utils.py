from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from common.models.api import ApiError
from common.task.executor import TaskPayload
from models.config.config import Config
from models.config.paths import ProjectPaths
from models.config.schema import TextualSchemaColumn

from models.topic.topic import TopicHierarchyModel, TopicModel

if TYPE_CHECKING:
  from sklearn.base import BaseEstimator
  from bertopic import BERTopic
  from networkx import DiGraph

@dataclass
class BERTopicColumnIntermediateResult:
  config: Config
  embedding_documents: list[str]
  documents: pd.Series
  # Marks which documents are excluded
  mask: pd.Series
  column: TextualSchemaColumn
  embeddings: np.ndarray
  task: TaskPayload
  model: BERTopic
  document_topic_assignments: list[int]
  
  topic_embeddings: np.ndarray

  @staticmethod
  def initialize(
    *,
    column: TextualSchemaColumn,
    config: Config,
    task: TaskPayload,
  )->"BERTopicColumnIntermediateResult":
    return BERTopicColumnIntermediateResult(
      column=column,
      config=config,
      task=task,
      hierarchy=None, # type: ignore
      document_topic_assignments=None, # type: ignore
      document_visualization_embeddings=None, # type: ignore
      topic_embeddings=None, # type: ignore
      topic_visualization_embeddings=None, # type: ignore
      documents=None, # type: ignore
      mask=None, # type: ignore
      embeddings=None, # type: ignore
      embedding_model=None, # type: ignore
      embedding_documents=None, # type: ignore
      model=None, # type: ignore
    )

def assert_valid_workspace_for_topic_modeling(df: pd.DataFrame, task: TaskPayload, config: Config):
  workspace_path = config.paths.full_path(ProjectPaths.Workspace)
  textual_columns = config.data_schema.textual()
  non_existent_columns = list(map(str, filter(lambda x: x not in df.columns, textual_columns)))
  if len(non_existent_columns) > 0:
    task.error(ApiError(f"{', '.join(non_existent_columns)} doesn't exist in the workspace. Please check your data source or column configuration to make sure that the column is present. Alternatively, delete \"{workspace_path}\" to fix any corrupted cached data.", 404))
