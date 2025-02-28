from dataclasses import dataclass, field
import functools
import threading
from typing import TYPE_CHECKING, cast
import pandas as pd

from modules.config import ProjectPaths, SchemaColumnTypeEnum, TextualSchemaColumn
from modules.logger import ProvisionedLogger
from modules.baseclass import Singleton
from modules.storage import CacheClient, CacheItem
from modules.topic import TopicModelingResult

from .config import Config
from .source import DataSource

if TYPE_CHECKING:
  from bertopic import BERTopic

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
  bertopic_models: CacheClient["BERTopic"] = field(
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
  
  def load_bertopic(self, column: str, *, no_cache: bool = False)->"BERTopic":
    from bertopic import BERTopic
    from modules.topic.bertopic_ext.builder import BERTopicModelBuilder

    cached_model = self.bertopic_models.get(column)
    textual_column = cast(TextualSchemaColumn, self.config.data_schema.assert_of_type(column, [SchemaColumnTypeEnum.Textual]))
    if cached_model is None or no_cache:
      model_path = self.config.paths.assert_path(ProjectPaths.BERTopic(column))
      embedding_model = BERTopicModelBuilder(self.id, textual_column, corpus_size=0).build_embedding_model()
      bertopic_model: BERTopic = BERTopic.load(model_path, embedding_model=embedding_model)
      self.bertopic_models.set(CacheItem(
        key=column,
        value=bertopic_model,
      ))
      return bertopic_model
    else:
      return cached_model
    
  def save_bertopic(self, model: "BERTopic", column:str):
    model_path = self.config.paths.assert_path(ProjectPaths.BERTopic(column))
    logger.info(f"Saved BERTopic model in \"{model_path}\".")
    model.save(model_path, "safetensors", save_ctfidf=True)
    self.bertopic_models.set(CacheItem(
      key=column,
      value=model,
    ))
  
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