from dataclasses import dataclass
import http
import os
import threading
from typing import TYPE_CHECKING, Optional
import numpy as np
import pandas as pd

from modules.api.wrapper import ApiError
from modules.project.cache_clients import BERTopicExperimentResultCacheAdapter, BERTopicModelCacheAdapter, ConfigCacheAdapter, DocumentEmbeddingsCacheAdapter, TopicEvaluationResultCacheAdapter, TopicModelingResultCacheAdapter, UMAPEmbeddingsCacheAdapter, VisualizationEmbeddingsCacheAdapter, WorkspaceCacheAdapter

from modules.config import Config, DataSource

from modules.logger import ProvisionedLogger
from modules.baseclass import Singleton
from modules.storage import CacheClient
from modules.storage.cache import CacheItem
from modules.topic.evaluation.model import TopicEvaluationResult
from modules.topic.experiments.model import BERTopicExperimentResult
from modules.topic.model import TopicModelingResult

if TYPE_CHECKING:
  from bertopic import BERTopic

logger = ProvisionedLogger().provision("ProjectCache")


@dataclass
class ProjectCache:
  id: str

  def __init__(self, project_id: str):
    self.id = project_id
    self.config_cache = ConfigCacheAdapter(
      project_id=project_id,
      cache=CacheClient[Config](
        name="Config", maxsize=1, ttl=5 * 60
      ),
    )
    self.workspaces = WorkspaceCacheAdapter(
      project_id=project_id,
      config=self.config_cache,
      cache=CacheClient[pd.DataFrame](
        name="Workspace", maxsize=20, ttl=5 * 60
      ),
    )
    self.topics = TopicModelingResultCacheAdapter(
      project_id=project_id,
      cache=CacheClient[TopicModelingResult](
        name="Topics", maxsize=5, ttl=5 * 60
      ),
    )
    self.bertopic_models = BERTopicModelCacheAdapter(
      project_id=project_id,
      config=self.config_cache,
      cache=CacheClient["BERTopic"](
        name="BERTopic Models", maxsize=5, ttl=5 * 60
      ),
    )
    self.document_vectors = DocumentEmbeddingsCacheAdapter(
      project_id=project_id,
      config=self.config_cache,
      workspace=self.workspaces,
      cache=CacheClient[np.ndarray](
        name="Document Vectors", maxsize=5, ttl=5 * 60
      ),
    )
    self.umap_vectors = UMAPEmbeddingsCacheAdapter(
      project_id=project_id,
      config=self.config_cache,
      workspace=self.workspaces,
      cache=CacheClient[np.ndarray](
        name="UMAP Vectors", maxsize=5, ttl=5 * 60
      ),
    )
    self.visualization_vectors = VisualizationEmbeddingsCacheAdapter(
      project_id=project_id,
      config=self.config_cache,
      workspace=self.workspaces,
      cache=CacheClient[np.ndarray](
        name="Visualization Vectors", maxsize=5, ttl=5 * 60
      ),
    )
    self.topic_evaluations = TopicEvaluationResultCacheAdapter(
      project_id=project_id,
      cache=CacheClient[TopicEvaluationResult](
        name="Topic Evaluation Results", maxsize=5, ttl=5 * 60
      ),
    )
    self.bertopic_experiments = BERTopicExperimentResultCacheAdapter(
      project_id=project_id,
      cache=CacheClient[BERTopicExperimentResult](
        name="BERTopic Experiment Results", maxsize=5, ttl=5 * 60
      ),
    )

  @property
  def config(self)->Config:
    return self.config_cache.load()
  
  def invalidate_topic_modeling(self, column: Optional[str]):
    self.topics.invalidate(key=column)
    self.bertopic_models.invalidate(key=column)
    self.document_vectors.invalidate(key=column)
    self.umap_vectors.invalidate(key=column)
    self.visualization_vectors.invalidate(key=column)
    self.topic_evaluations.invalidate(key=column)
    self.bertopic_experiments.invalidate(key=column)
    
  def invalidate(self):
    self.config_cache.invalidate()
    self.workspaces.invalidate()
    self.invalidate_topic_modeling(column=None)

class DataSourceCacheManager(metaclass=Singleton):
  cache: CacheClient[pd.DataFrame]
  def __init__(self):
    self.cache = CacheClient(name="Data Source", maxsize=2, ttl=2 * 60 * 1000)

def get_cached_data_source(source: "DataSource", with_cache: bool = True):
  import hashlib
  cache_key = hashlib.md5(str(source).encode(encoding='utf-8')).hexdigest()
  cache = DataSourceCacheManager().cache

  cached_dataframe = cache.get(cache_key)
  if cached_dataframe is not None and with_cache:
    logger.info(f"Loaded data source {source.path} from Cache {cache.name}")
    return cached_dataframe
  
  if not os.path.exists(source.path):
    raise ApiError(f"We were unable to find any dataset files in \"{source.path}\". Please check your dataset path again and ensure that it exists.", http.HTTPStatus.NOT_FOUND)
  try:
    source_dataframe = source.load()
    logger.info(f"Loaded data source from {source}")
    cache.set(CacheItem(
      value=source_dataframe,
      key=cache_key,
    ))
    return source_dataframe
  except Exception as e:
    raise ApiError(f"The dataset in \"{source.path}\" cannot be loaded due to the following reason: {str(e)}. Please make sure that your dataset file is a valid {source.type.upper()} file.", http.HTTPStatus.UNPROCESSABLE_ENTITY)

__all__ = [
  "ProjectCache",
  "get_cached_data_source"
]