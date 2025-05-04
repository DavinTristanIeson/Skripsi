import abc
from dataclasses import dataclass
from http import HTTPStatus
import os
from typing import TYPE_CHECKING, Generic, Optional, TypeVar, cast

import numpy as np
import pandas as pd
from pydantic import ValidationError
from modules.api.wrapper import ApiError
from modules.config.config import Config
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.logger.provisioner import ProvisionedLogger
from modules.project.paths import ProjectPathManager, ProjectPaths
from modules.storage.cache import CacheClient, CacheItem
from modules.topic.bertopic_ext.dimensionality_reduction import VisualizationCachedUMAP
from modules.topic.evaluation.model import TopicEvaluationResult
from modules.topic.experiments.model import BERTopicExperimentResult
from modules.topic.model import TopicModelingResult

if TYPE_CHECKING:
  from bertopic import BERTopic
  
logger = ProvisionedLogger().provision("ProjectCache")

T = TypeVar("T")

@dataclass
class ProjectCacheAdapter(Generic[T], abc.ABC):
  project_id: str
  cache: CacheClient[T]

  @abc.abstractmethod
  def _save(self, value: T, key: str)->Optional[CacheItem[T]]:
    ...

  @abc.abstractmethod
  def _load(self, key: str)->T | CacheItem[T]:
    ...

  def save(self, value: T, key: str)->None:
    cached_item = self._save(value, key)
    if cached_item is not None:
      self.cache.set(cached_item)
    else:
      self.cache.set(CacheItem(
        key=key,
        value=value
      ))

  def load(self, key: str)->T:
    cached_value = self.cache.get(key)
    if cached_value is not None:
      return cached_value
    loaded_value = self._load(key)
    if isinstance(loaded_value, CacheItem):
      self.cache.set(loaded_value)
      return loaded_value.value
    else:
      self.cache.set(CacheItem(
        key=key,
        value=loaded_value,
      ))
      return loaded_value
    
  def clear(self):
    self.cache.clear()

@dataclass
class ConfigCacheAdapter:
  project_id: str
  cache: CacheClient[Config]
  def save(self, config: Config)->None:
    self.cache.set(CacheItem(
      key=self.project_id,
      value=config,
      persistent=True,
    ))
    config.save_to_json()

  def load(self)->Config:
    cached_config = self.cache.get(self.project_id)
    if cached_config is not None:
      return cached_config
    config = Config.from_project(self.project_id)
    self.cache.set(CacheItem(
      key=self.project_id,
      value=config,
      persistent=True,
    ))
    return config
  
  def clear(self):
    self.cache.clear()
  
@dataclass
class WorkspaceCacheAdapter:
  project_id: str
  cache: CacheClient[pd.DataFrame]
  config: ConfigCacheAdapter
  
  def set(self, df: pd.DataFrame, key: str):
    self.cache.set(CacheItem(
      key=key,
      value=df
    ))

  def get(self, key: str)->Optional[pd.DataFrame]:
    return self.cache.get(key)

  def load(self)->pd.DataFrame:
    empty_key = ''
    cached_df = self.cache.get(empty_key)
    if cached_df is not None:
      return cached_df
    
    config = self.config.load()
    df = config.load_workspace()
    self.cache.set(CacheItem(
      key=empty_key,
      value=df,
      persistent=True
    ))
    return df
  
  def save(self, df: pd.DataFrame):
    config = self.config.load()
    config.save_workspace(df)
    self.cache.clear()
    self.cache.set(CacheItem(
      key='',
      persistent=True,
      value=df
    ))

  def clear(self):
    self.cache.clear()

class TopicModelingResultCacheAdapter(ProjectCacheAdapter[TopicModelingResult]):
  def _save(self, value, key):
    value.save_as_json(key)

  def _load(self, key):
    return TopicModelingResult.load(self.project_id, key)
 
@dataclass
class BERTopicModelCacheAdapter(ProjectCacheAdapter["BERTopic"]):
  config: ConfigCacheAdapter
  def _load(self, key):
    from bertopic import BERTopic
    from modules.topic.bertopic_ext.builder import BERTopicModelBuilder

    config = self.config.load()
    column = key
    textual_column = cast(
      TextualSchemaColumn,
      config.data_schema.assert_of_type(column, [SchemaColumnTypeEnum.Textual])
    )
    
    model_path = config.paths.assert_path(ProjectPaths.BERTopic(column))
    embedding_model = BERTopicModelBuilder(
      project_id=self.project_id,
      column=textual_column,
      corpus_size=0,
    ).build_embedding_model()
    bertopic_model: BERTopic = BERTopic.load(model_path, embedding_model=embedding_model)
    return bertopic_model
    
  def _save(self, value, key):
    config = self.config.load()
    model_path = config.paths.allocate_path(ProjectPaths.BERTopic(key))
    value.save(model_path, "safetensors", save_ctfidf=True)

@dataclass
class VisualizationEmbeddingsCacheAdapter(ProjectCacheAdapter[np.ndarray]):
  config: ConfigCacheAdapter
  def __prepare(self, key: str):
    config = self.config.load()
    textual_column = cast(
      TextualSchemaColumn,
      config.data_schema.assert_of_type(key, [SchemaColumnTypeEnum.Textual])
    )
    visumap = VisualizationCachedUMAP(
      project_id=self.project_id,
      column=textual_column,
      low_memory=True
    )
    return visumap
  
  def _load(self, key):
    visumap = self.__prepare(key)
    visualization_vectors = visumap.load_cached_embeddings()
    if visualization_vectors is None:
      raise ApiError(f"There are no document/topic embeddings for \"{visumap.column.name}\". The file may be corrupted or the topic modeling procedure has not been executed on this column.", HTTPStatus.BAD_REQUEST)
    return visualization_vectors
  
  def _save(self, value, key):
    visumap = self.__prepare(key)
    visumap.save_embeddings(value)

@dataclass
class TopicEvaluationResultCacheAdapter(ProjectCacheAdapter[TopicEvaluationResult]):
  def _load(self, key):
    paths = ProjectPathManager(project_id=self.project_id)
    file_path = paths.full_path(ProjectPaths.TopicEvaluation(key))
    if not os.path.exists(file_path):
      raise ApiError(f"There are no cached topic evaluation results in \"{file_path}\". It seems that you have not performed any evaluations on the topics of \"{key}\".", HTTPStatus.NOT_FOUND)
    
    with open(file_path, 'r', encoding='utf-8') as f:
      try:
        result = TopicEvaluationResult.model_validate_json(f.read())
      except ValidationError:
        raise ApiError(f"We were unable to read the topic evaluation results in \"{file_path}\". The file may have been corrupted.", HTTPStatus.UNPROCESSABLE_ENTITY)
      
    self.cache.set(CacheItem(
      key=key,
      value=result,
    ))
    return result
    
  def _save(self, value, key):
    paths = ProjectPathManager(project_id=self.project_id)
    file_path = paths.allocate_path(ProjectPaths.TopicEvaluation(key))
    with open(file_path, 'w', encoding='utf-8') as f:
      f.write(value.model_dump_json())


@dataclass
class BERTopicExperimentResultCacheAdapter(ProjectCacheAdapter[BERTopicExperimentResult]):
  def _load(self, key):
    paths = ProjectPathManager(project_id=self.project_id)
    file_path = paths.full_path(ProjectPaths.TopicExperiments(key))
    if not os.path.exists(file_path):
      raise ApiError(f"There are no cached BERTopic experiment results in \"{file_path}\". It seems that you have not performed any BERTopic experiments.", HTTPStatus.NOT_FOUND)
    
    with open(file_path, 'r', encoding='utf-8') as f:
      try:
        result = BERTopicExperimentResult.model_validate_json(f.read())
      except ValidationError:
        raise ApiError(f"We were unable to read the BERTopic experiment results in \"{file_path}\". The file may have been corrupted.", HTTPStatus.UNPROCESSABLE_ENTITY)
      
    self.cache.set(CacheItem(
      key=key,
      value=result,
    ))
    return result
    
  def _save(self, value, key):
    paths = ProjectPathManager(project_id=self.project_id)
    file_path = paths.allocate_path(ProjectPaths.TopicExperiments(key))
    with open(file_path, 'w', encoding='utf-8') as f:
      f.write(value.model_dump_json())
