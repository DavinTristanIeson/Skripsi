from dataclasses import dataclass, field
import functools
import itertools
import threading
from typing import Annotated, Sequence

from fastapi import Depends
from common.logger import RegisteredLogger
from common.models.metaclass import Singleton
from common.storage.cache import CacheClient
from models.config import Config
import pandas as pd

from models.config.source import DataSource
from models.table.filter import TableSort
from models.table.filter_variants import TableFilter

logger = RegisteredLogger().provision("CacheClient")

@dataclass
class ProjectCache:
  id: str
  config: Config
  workspaces: CacheClient[pd.DataFrame] = field(
    default_factory=lambda: CacheClient(name="Workspace", maxsize=5, ttl=10 * 60),
    init=False,
  )

  @classmethod
  def workspace_key(cls, filters: Sequence[TableFilter], sorts: Sequence[TableSort]):
    if len(filters) == 0:
      return ''
    return ' '.join([hex(hash(filter)) for filter in itertools.chain(filters, sorts)])


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

def get_cached_project(project_id: str):
  return ProjectCacheManager().get(project_id)

@functools.lru_cache(2)
def get_cached_data_source(source: DataSource):
  logger.info(f"Loaded data source from {source}")
  return source.load()

ProjectCacheDependency = Annotated[ProjectCache, Depends(get_cached_project)]

__all__ = [
  "ProjectCacheManager",
  "ProjectCacheDependency",
  "get_cached_data_source"
]