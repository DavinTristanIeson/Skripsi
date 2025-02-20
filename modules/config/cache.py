from dataclasses import dataclass, field
import functools
import threading
from typing import Annotated
import pandas as pd

from fastapi import Depends
from modules.logger import ProvisionedLogger
from modules.baseclass import Singleton
from modules.storage.cache import CacheClient, CacheItem

from .config import Config
from .source import DataSource

logger = ProvisionedLogger().provision("CacheClient")

@dataclass
class ProjectCache:
  id: str
  config: Config
  workspaces: CacheClient[pd.DataFrame] = field(
    default_factory=lambda: CacheClient(name="Workspace", maxsize=5, ttl=10 * 60),
    init=False,
  )

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
          config=Config.from_project(project_id)
        )
        self.projects[project_id] = cache
      return cache

  def invalidate(self, project_id: str):
    with self.lock:
      self.projects.pop(project_id, None)

def __get_cached_project(project_id: str):
  return ProjectCacheManager().get(project_id)

@functools.lru_cache(2)
def get_cached_data_source(source: DataSource):
  logger.info(f"Loaded data source from {source}")
  return source.load()

ProjectCacheDependency = Annotated[ProjectCache, Depends(__get_cached_project)]

__all__ = [
  "ProjectCacheManager",
  "ProjectCacheDependency",
  "get_cached_data_source"
]