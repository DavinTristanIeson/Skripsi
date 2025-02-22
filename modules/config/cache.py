from dataclasses import dataclass, field
import functools
import threading
from typing import Annotated
import pandas as pd

from fastapi import Depends
from modules.logger import ProvisionedLogger
from modules.baseclass import Singleton
from modules.storage.cache import CacheClient, CacheItem
from modules.topic.model import TopicModelingResult

from .config import Config
from .source import DataSource

logger = ProvisionedLogger().provision("CacheClient")

@dataclass
class ProjectCache:
  id: str
  workspaces: CacheClient[pd.DataFrame] = field(
    default_factory=lambda: CacheClient(name="Workspace", maxsize=5, ttl=10 * 60),
    init=False,
  )
  topics: CacheClient[TopicModelingResult] = field(
    default_factory=lambda: CacheClient(name="Topics", maxsize=None, ttl=None),
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
    else:
      return cached_topic

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
def get_cached_data_source(source: DataSource):
  logger.info(f"Loaded data source from {source}")
  return source.load()


__all__ = [
  "ProjectCacheManager",
  "ProjectCache",
  "get_cached_data_source"
]