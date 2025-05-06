from dataclasses import dataclass, field
import functools
import http
import os
import threading
from typing import TYPE_CHECKING, cast
import numpy as np
import pandas as pd

from modules.api.wrapper import ApiError
from modules.project.cache_clients import BERTopicExperimentResultCacheAdapter, BERTopicModelCacheAdapter, ConfigCacheAdapter, TopicEvaluationResultCacheAdapter, TopicModelingResultCacheAdapter, VisualizationEmbeddingsCacheAdapter, WorkspaceCacheAdapter
from modules.project.lock import ProjectLockManager
from modules.project.paths import ProjectPathManager, ProjectPaths

from modules.config import SchemaColumnTypeEnum, TextualSchemaColumn, Config, DataSource

from modules.logger import ProvisionedLogger
from modules.baseclass import Singleton
from modules.storage import CacheClient, CacheItem
from modules.topic.bertopic_ext.dimensionality_reduction import VisualizationCachedUMAP
from modules.topic.evaluation.model import TopicEvaluationResult
from modules.topic.experiments.model import BERTopicExperimentResult
from modules.topic.model import TopicModelingResult

if TYPE_CHECKING:
  from bertopic import BERTopic

logger = ProvisionedLogger().provision("ProjectCache")


@dataclass
class ProjectCache:
  id: str
  lock: threading.RLock

  def __init__(self, project_id: str, lock: threading.RLock):
    self.id = project_id
    self.lock = lock
    self.config_cache = ConfigCacheAdapter(
      project_id=project_id,
      lock=lock,
      cache=CacheClient[Config](
        name="Config", maxsize=1, ttl=5 * 60
      ),
    )
    self.workspaces = WorkspaceCacheAdapter(
      project_id=project_id,
      lock=lock,
      config=self.config_cache,
      cache=CacheClient[pd.DataFrame](
        name="Workspace", maxsize=20, ttl=5 * 60
      ),
    )
    self.topics = TopicModelingResultCacheAdapter(
      project_id=project_id,
      lock=lock,
      cache=CacheClient[TopicModelingResult](
        name="Topics", maxsize=5, ttl=5 * 60
      ),
    )
    self.bertopic_models = BERTopicModelCacheAdapter(
      project_id=project_id,
      lock=lock,
      config=self.config_cache,
      cache=CacheClient["BERTopic"](
        name="BERTopic Models", maxsize=5, ttl=5 * 60
      ),
    )
    self.visualization_vectors = VisualizationEmbeddingsCacheAdapter(
      project_id=project_id,
      lock=lock,
      config=self.config_cache,
      cache=CacheClient[np.ndarray](
        name="Visualization Embeddings", maxsize=5, ttl=5 * 60
      ),
    )
    self.topic_evaluations = TopicEvaluationResultCacheAdapter(
      project_id=project_id,
      lock=lock,
      cache=CacheClient[TopicEvaluationResult](
        name="Topic Evaluation Results", maxsize=5, ttl=5 * 60
      ),
    )
    self.bertopic_experiments = BERTopicExperimentResultCacheAdapter(
      project_id=project_id,
      lock=lock,
      cache=CacheClient[BERTopicExperimentResult](
        name="BERTopic Experiment Results", maxsize=5, ttl=5 * 60
      ),
    )

  @property
  def config(self)->Config:
    return self.config_cache.load()
    
  def invalidate(self):
    with self.lock:
      self.config_cache.invalidate()
      self.workspaces.invalidate()
      self.topics.invalidate()
      self.bertopic_models.invalidate()
      self.visualization_vectors.invalidate()
      self.topic_evaluations.invalidate()
      self.bertopic_experiments.invalidate()

class ProjectCacheManager(metaclass=Singleton):
  projects: dict[str, ProjectCache]
  lock: threading.RLock
  def __init__(self):
    super().__init__()
    self.projects = dict()
    self.lock = threading.RLock()

  def get(self, project_id: str)->ProjectCache:
    with self.lock:
      cache = self.projects.get(project_id, None)
      if cache is None:
        cache = ProjectCache(
          project_id=project_id,
          lock=ProjectLockManager().get(project_id),
        )
    self.projects[project_id] = cache
    return cache

  def invalidate(self, project_id: str):
    project_cache = self.projects.get(project_id)
    if project_cache is not None:
      project_cache.invalidate()

@functools.lru_cache(2)
def get_cached_data_source(source: "DataSource"):
  logger.info(f"Loaded data source from {source}")
  if not os.path.exists(source.path):
    raise ApiError(f"We were unable to find any dataset files in \"{source.path}\". Please check your dataset path again and ensure that it exists.", http.HTTPStatus.NOT_FOUND)
  try:
    return source.load()
  except Exception as e:
    raise ApiError(f"The dataset in \"{source.path}\" cannot be loaded due to the following reason: {str(e)}. Please make sure that your dataset file is a valid {source.type.upper()} file.", http.HTTPStatus.UNPROCESSABLE_ENTITY)

__all__ = [
  "ProjectCacheManager",
  "ProjectCache",
  "get_cached_data_source"
]