from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from common.models.api import ApiError
from common.task.executor import TaskPayload
from models.config.config import Config
from models.config.paths import ProjectPaths
from models.config.schema import TextualSchemaColumn
import numpy.typing as npt

if TYPE_CHECKING:
  from sklearn.base import BaseEstimator

@dataclass
class BERTopicColumnIntermediateResult:
  config: Config
  embedding_documents: list[str]
  documents: list[str]
  # Marks which documents are excluded
  mask: pd.Series
  column: TextualSchemaColumn
  embeddings: npt.NDArray
  embedding_model: BaseEstimator
  task: TaskPayload

def assert_valid_workspace_for_topic_modeling(df: pd.DataFrame, task: TaskPayload, config: Config):
  workspace_path = config.paths.full_path(ProjectPaths.Workspace)
  textual_columns = config.data_schema.textual()
  non_existent_columns = list(map(str, filter(lambda x: x not in df.columns, textual_columns)))
  if len(non_existent_columns) > 0:
    task.error(ApiError(f"{', '.join(non_existent_columns)} doesn't exist in the workspace. Please check your data source or column configuration to make sure that the column is present. Alternatively, delete \"{workspace_path}\" to fix any corrupted cached data.", 404))
