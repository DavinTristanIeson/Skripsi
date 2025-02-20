from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from modules.api import ApiError
from modules.task import TaskPayload
from modules.config import Config, ProjectPaths, TextualSchemaColumn

if TYPE_CHECKING:
  from bertopic import BERTopic

@dataclass
class _BERTopicColumnIntermediateResult:
  config: Config
  embedding_documents: list[str]
  documents: pd.Series
  # Marks which documents are excluded
  mask: pd.Series
  column: TextualSchemaColumn
  embeddings: np.ndarray
  task: TaskPayload
  model: "BERTopic"
  document_topic_assignments: np.ndarray
  
  @staticmethod
  def initialize(
    *,
    column: TextualSchemaColumn,
    config: Config,
    task: TaskPayload,
  )->"_BERTopicColumnIntermediateResult":
    return _BERTopicColumnIntermediateResult(
      column=column,
      config=config,
      task=task,
      document_topic_assignments=None, # type: ignore
      documents=None, # type: ignore
      mask=None, # type: ignore
      embeddings=None, # type: ignore
      embedding_documents=None, # type: ignore
      model=None, # type: ignore
    )

def assert_valid_workspace_for_topic_modeling(df: pd.DataFrame, task: TaskPayload, config: Config):
  workspace_path = config.paths.full_path(ProjectPaths.Workspace)
  textual_columns = config.data_schema.textual()
  non_existent_columns = list(map(str, filter(lambda x: x.name not in df.columns, textual_columns)))
  if len(non_existent_columns) > 0:
    raise ApiError(f"{', '.join(non_existent_columns)} doesn't exist in the workspace. Please check your data source or column configuration to make sure that the column is present. Alternatively, delete \"{workspace_path}\" to fix any corrupted cached data.", 404)

__all__ = [
  "assert_valid_workspace_for_topic_modeling"
]