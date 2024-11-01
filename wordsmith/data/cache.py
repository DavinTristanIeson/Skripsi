from dataclasses import field, dataclass
from typing import Callable, Generic, Optional, TypeVar

import pandas as pd
from common.models.metaclass import Singleton
from wordsmith.data.config import Config
from wordsmith.data.paths import ProjectPathManager


T = TypeVar("T")

@dataclass
class ProjectDependencyCache(Generic[T]):
  max_size: int
  factory: Callable[[str], T]
  items: dict[str, T] = field(default_factory=lambda: dict())

  def cleanup(self):
    if len(self.items) <= self.max_size:
      return

    difference = len(self.items) - self.max_size
    for key, idx in zip(iter(self.items), range(difference)):
      self.items.pop(key)

  def get(self, project_id: str)->T:
    item = self.items.get(project_id, None)
    if item is not None:
      return item
    
    item = self.factory(project_id)
    self.items[project_id] = item
    self.cleanup()
    
    return item
  
  def invalidate(self, project_id: str)->Optional[T]:
    if project_id in self.items:
      return self.items.pop(project_id)
    return None



class ProjectCacheManager(metaclass=Singleton):
  workspaces: ProjectDependencyCache[pd.DataFrame]
  configs: ProjectDependencyCache[Config]

  def __init__(self):
    self.workspaces = ProjectDependencyCache(
      max_size=4,
      factory=lambda project_id: ProjectPathManager(project_id=project_id).load_workspace(),
    )
    self.configs = ProjectDependencyCache(
      max_size=16,
      factory=lambda project_id: Config.from_project(project_id),
    )