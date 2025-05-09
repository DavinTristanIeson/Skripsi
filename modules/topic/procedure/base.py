import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from modules.project.cache import ProjectCache
from modules.project.cache_manager import ProjectCacheManager
from modules.task import TaskManagerProxy
from modules.config import Config, TextualSchemaColumn
from modules.topic.model import TopicModelingResult

if TYPE_CHECKING:
  from bertopic import BERTopic


class BERTopicIntermediateState:
  config: Config
  column: TextualSchemaColumn
  
  @property
  def cache(self)->ProjectCache:
    return ProjectCacheManager().get(self.config.project_id)

  # Configured BERTopic model
  model: "BERTopic"
  # Documents used for embedding
  embedding_documents: list[str]
  # Documents used for c-TF-IDF
  documents: pd.Series
  # Marks which documents are excluded from the original dataset
  mask: pd.Series

  # Document vectors
  document_vectors: np.ndarray

  # The topic assignments for each topic
  document_topic_assignments: np.ndarray

  result: TopicModelingResult
  
@dataclass
class BERTopicProcedureComponent(abc.ABC):
  state: BERTopicIntermediateState
  task: TaskManagerProxy
  @abc.abstractmethod
  def run(self):
    pass

__all__ = [
  "BERTopicIntermediateState",
  "BERTopicProcedureComponent"
]