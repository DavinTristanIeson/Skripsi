from dataclasses import dataclass, field
import functools
import http
import threading
from typing import TYPE_CHECKING, cast
import numpy as np
import pandas as pd

from modules.api.wrapper import ApiError
from modules.project.paths import ProjectPathManager, ProjectPaths

from modules.config import SchemaColumnTypeEnum, TextualSchemaColumn, Config, DataSource

from modules.logger import ProvisionedLogger
from modules.baseclass import Singleton
from modules.storage import CacheClient, CacheItem
from modules.topic.bertopic_ext.dimensionality_reduction import VisualizationCachedUMAP
from modules.topic.model import TopicModelingResult

if TYPE_CHECKING:
  from bertopic import BERTopic

logger = ProvisionedLogger().provision("CacheClient")


@dataclass
class ProjectCache:
  id: str
  workspaces: CacheClient[pd.DataFrame] = field(
    default_factory=lambda: CacheClient(name="Workspace", maxsize=20, ttl=5 * 60),
    init=False,
  )
  topics: CacheClient[TopicModelingResult] = field(
    default_factory=lambda: CacheClient(name="Topics", maxsize=None, ttl=5 * 60),
    init=False,
  )
  bertopic_models: CacheClient["BERTopic"] = field(
    default_factory=lambda: CacheClient(name="BERTopic Models", maxsize=5, ttl=5 * 60),
    init=False,
  )
  visualization_vectors: CacheClient[np.ndarray] = field(
    default_factory=lambda: CacheClient(name="Visualization Vectors", maxsize=5, ttl=5 * 60),
    init=False,
  )

  @functools.cached_property
  def config(self):
    return Config.from_project(self.id)
  
  def load_topic(self, column: str):
    cached_topic = self.topics.get(column)
    if cached_topic is None:
      topic_result = TopicModelingResult.load(self.id, column)
      self.topics.set(CacheItem(
        key=column,
        value=topic_result,
      ))
      return topic_result
    else:
      return cached_topic
    
  def save_topic(self, tm_result: TopicModelingResult, column: str):
    tm_result.save_as_json(column)
    # reinitialize cache
    self.topics.set(CacheItem(
      key=column,
      persistent=True,
      value=tm_result
    ))

  def load_workspace(self)->pd.DataFrame:
    empty_key = ''
    cached_df = self.workspaces.get(empty_key)
    if cached_df is not None:
      return cached_df
    
    df = self.config.load_workspace()
    self.workspaces.set(CacheItem(
      key=empty_key,
      value=df,
      persistent=True
    ))
    return df
  
  def save_workspace(self, df: pd.DataFrame):
    self.config.save_workspace(df)
    self.workspaces.clear()
    # reinitialize cache
    self.workspaces.set(CacheItem(
      key='',
      persistent=True,
      value=df
    ))
  
  def load_bertopic(self, column: str)->"BERTopic":
    from bertopic import BERTopic
    from modules.topic.bertopic_ext.builder import BERTopicModelBuilder

    cached_model = self.bertopic_models.get(column)
    textual_column = cast(TextualSchemaColumn, self.config.data_schema.assert_of_type(column, [SchemaColumnTypeEnum.Textual]))
    if cached_model is not None:
      return cached_model
    model_path = self.config.paths.assert_path(ProjectPaths.BERTopic(column))
    embedding_model = BERTopicModelBuilder(self.id, textual_column, corpus_size=0).build_embedding_model()
    bertopic_model: BERTopic = BERTopic.load(model_path, embedding_model=embedding_model)
    self.bertopic_models.set(CacheItem(
      key=column,
      value=bertopic_model,
    ))
    return bertopic_model
    
  def save_bertopic(self, model: "BERTopic", column:str):
    model_path = self.config.paths.allocate_path(ProjectPaths.BERTopic(column))
    model.save(model_path, "safetensors", save_ctfidf=True)
    logger.info(f"Saved BERTopic model in \"{model_path}\".")
    self.bertopic_models.set(CacheItem(
      key=column,
      value=model,
    ))

  def load_visualization_vectors(self, column: TextualSchemaColumn)->np.ndarray:
    cached_visualization_vectors = self.visualization_vectors.get(column.name)
    if cached_visualization_vectors is not None:
      return cached_visualization_vectors

    config = self.config
    visumap = VisualizationCachedUMAP(
      column=column,
      low_memory=True,
      project_id=config.project_id,
    )

    visualization_vectors = visumap.load_cached_embeddings()
    if visualization_vectors is None:
      raise ApiError(f"There are no document/topic embeddings for \"{column.name}\". The file may be corrupted or the topic modeling procedure has not been executed on this column.", http.HTTPStatus.BAD_REQUEST)
   
    self.visualization_vectors.set(CacheItem(
      key=column.name,
      value=visualization_vectors,
    ))
    return visualization_vectors
  
  
class ProjectCacheManager(metaclass=Singleton):
  projects: dict[str, ProjectCache]
  lock: threading.Lock
  def __init__(self):
    super().__init__()
    self.projects = dict()
    self.lock = threading.Lock()

  def get(self, project_id: str)->ProjectCache:
    with self.lock:
      cache = self.projects.get(project_id, None)
      if cache is None:
        cache = ProjectCache(
          id=project_id,
        )
        self.projects[project_id] = cache
      return cache

  def invalidate(self, project_id: str):
    with self.lock:
      self.projects.pop(project_id, None)

@functools.lru_cache(2)
def get_cached_data_source(source: "DataSource"):
  logger.info(f"Loaded data source from {source}")
  return source.load()


__all__ = [
  "ProjectCacheManager",
  "ProjectCache",
  "get_cached_data_source"
]