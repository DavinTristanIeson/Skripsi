import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Optional, TypeVar, cast

import numpy as np
import pandas as pd
from pydantic import ValidationError
from modules.config.config import Config
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.exceptions.files import CorruptedFileException, FileLoadingException, FileNotExistsException
from modules.logger.provisioner import ProvisionedLogger
from modules.project.lock import ProjectFileLockManager
from modules.project.paths import ProjectPathManager, ProjectPaths
from modules.storage.atomic import atomic_write
from modules.storage.cache import CacheClient, CacheItem
from modules.storage.transformer import CachedEmbeddingBehavior
from modules.topic.bertopic_ext.dimensionality_reduction import BERTopicCachedUMAP, VisualizationCachedUMAP
from modules.topic.bertopic_ext.embedding import BERTopicEmbeddingModelFactory
from modules.topic.evaluation.model import TopicEvaluationResult
from modules.topic.exceptions import MissingCachedTopicModelingResult, UnsyncedDocumentVectorsException
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
    logger.debug(f"CACHE ({self.cache.name}): SAVE {key}")
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
    logger.debug(f"CACHE ({self.cache.name}): LOAD {key}")
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
    
  def invalidate(self, key: Optional[str] = None, prefix: Optional[str] = None):
    self.cache.invalidate(key=key, prefix=prefix)
  
@dataclass
class ConfigCacheAdapter:
  project_id: str
  cache: CacheClient[Config]
  @property
  def lock(self):
    return ProjectFileLockManager().lock_file(
      project_id=self.project_id,
      path=ProjectPaths.Config,
      wait=True,
    )
  
  def save(self, config: Config)->None:
    logger.debug(f"CACHE ({self.cache.name}): SAVE CONFIG")
    with self.lock:
      config.save_to_json()
    self.cache.set(CacheItem(
      key=self.project_id,
      value=config,
      persistent=True,
    ))

  def load(self)->Config:
    cached_config = self.cache.get(self.project_id)
    if cached_config is not None:
      return cached_config
    logger.debug(f"CACHE ({self.cache.name}): LOAD CONFIG")
    with self.lock:
      config = Config.from_project(self.project_id)
    self.cache.set(CacheItem(
      key=self.project_id,
      value=config,
      persistent=True,
    ))
    return config
  
  def invalidate(self):
    self.cache.clear()
  
@dataclass
class WorkspaceCacheAdapter:
  project_id: str
  cache: CacheClient[pd.DataFrame]
  config: ConfigCacheAdapter

  @property
  def lock(self):
    return ProjectFileLockManager().lock_file(
      project_id=self.project_id,
      path=ProjectPaths.Workspace,
      wait=True,
    )
  
  def set(self, df: pd.DataFrame, key: str):
    self.cache.set(CacheItem(
      key=key,
      value=df
    ))

  def get(self, key: str)->Optional[pd.DataFrame]:
    return self.cache.get(key)

  def load(self, *, cached: bool = True)->pd.DataFrame:
    empty_key = ''
    cached_df = self.cache.get(empty_key)
    if cached_df is not None and cached:
      return cached_df
    logger.debug(f"CACHE ({self.cache.name}): LOAD WORKSPACE")
    
    config = self.config.load()
    with self.lock:
      df = config.load_workspace()
    self.cache.set(CacheItem(
      key=empty_key,
      value=df,
      persistent=True
    ))
    return df
  
  def save(self, df: pd.DataFrame):
    config = self.config.load()
    logger.debug(f"CACHE ({self.cache.name}): SAVE WORKSPACE")
    with self.lock:
      config.save_workspace(df)
    self.cache.clear()
    self.cache.set(CacheItem(
      key='',
      persistent=True,
      value=df
    ))

  def invalidate(self):
    self.cache.clear()

class TopicModelingResultCacheAdapter(ProjectCacheAdapter[TopicModelingResult]):
  def _save(self, value, key):
    value.save_as_json(key)

  def _load(self, key):
    return TopicModelingResult.load(self.project_id, key)
 
@dataclass
class BERTopicModelCacheAdapter(ProjectCacheAdapter["BERTopic"]):
  config: ConfigCacheAdapter

  def lock(self, key: str):
    return ProjectFileLockManager().lock_file(
      project_id=self.project_id,
      path=ProjectPaths.BERTopic(key),
      wait=True,
    )
  
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
    try:
      with self.lock(key):
        bertopic_model: BERTopic = BERTopic.load(model_path, embedding_model=embedding_model)
    except Exception:
      raise CorruptedFileException(
        CorruptedFileException.format_message(
          path=model_path,
          purpose="BERTopic model",
          solution="Try running the topic modeling algorithm again."
        )
      )
    return bertopic_model
    
  def _save(self, value, key):
    config = self.config.load()
    model_path = config.paths.allocate_path(ProjectPaths.BERTopic(key))
    with self.lock(key):
      value.save(model_path, "safetensors", save_ctfidf=True)


@dataclass
class GenericEmbeddingsCacheAdapter(ProjectCacheAdapter[np.ndarray], abc.ABC):
  config: ConfigCacheAdapter
  workspace: WorkspaceCacheAdapter

  @abc.abstractmethod
  def _get_file_path(self, column: str)->str:
    ...

  def lock(self, key: str):
    return ProjectFileLockManager().lock_file(
      project_id=self.project_id,
      path=self._get_file_path(key),
      wait=True,
    )
  
  @abc.abstractmethod
  def _get_cached_model(self, column: TextualSchemaColumn)->CachedEmbeddingBehavior:
    ...

  @abc.abstractmethod
  def _get_label(self)->str:
    ...
  
  def _prepare(self, key: str):
    config = self.config.load()
    textual_column = cast(
      TextualSchemaColumn,
      config.data_schema.assert_of_type(key, [SchemaColumnTypeEnum.Textual])
    )
    return self._get_cached_model(textual_column), textual_column
  
  def _load(self, key):
    cached_model, textual_column = self._prepare(key)
    with self.lock(key):
      cached_vectors = cached_model.load_cached_embeddings()
    df = self.workspace.load()
    documents_column = df[textual_column.preprocess_column.name]
    mask = documents_column.notna() & (documents_column.str.len() > 0)
    corpus_size = len(df[mask])
    if cached_vectors is None:
      raise MissingCachedTopicModelingResult(
        type=self._get_label(),
        column=key,
      )
    if len(cached_vectors) != corpus_size:
      raise UnsyncedDocumentVectorsException(
        type=self._get_label(),
        expected_rows=corpus_size,
        observed_rows=len(cached_vectors),
        column=key,
      )

    return cached_vectors
  
  def _save(self, value, key):
    cached_model, textual_column = self._prepare(key)
    cached_model.save_embeddings(value)


@dataclass
class DocumentEmbeddingsCacheAdapter(GenericEmbeddingsCacheAdapter):
  def _get_file_path(self, column: str)->str:
    return ProjectPaths.DocumentEmbeddings(column)

  def _get_cached_model(self, column)->CachedEmbeddingBehavior:
    return BERTopicEmbeddingModelFactory(
      project_id=self.project_id,
      column=column
    ).build()

  def _get_label(self)->str:
    return "document vectors"
  
@dataclass
class UMAPEmbeddingsCacheAdapter(GenericEmbeddingsCacheAdapter):
  def _get_file_path(self, column: str)->str:
    return ProjectPaths.UMAPEmbeddings(column)

  def _get_cached_model(self, column)->CachedEmbeddingBehavior:
    return BERTopicCachedUMAP(
      project_id=self.project_id,
      column=column,
      low_memory=True
    )

  def _get_label(self)->str:
    return "UMAP vectors"

@dataclass
class VisualizationEmbeddingsCacheAdapter(GenericEmbeddingsCacheAdapter):
  def _get_file_path(self, column: str)->str:
    return ProjectPaths.VisualizationEmbeddings(column)

  def _get_cached_model(self, column)->CachedEmbeddingBehavior:
    return VisualizationCachedUMAP(
      project_id=self.project_id,
      column=column,
      low_memory=True
    )

  def _get_label(self)->str:
    return "visualization vectors"

@dataclass
class TopicEvaluationResultCacheAdapter(ProjectCacheAdapter[TopicEvaluationResult]):
  def lock(self, key: str):
    return ProjectFileLockManager().lock_file(
      project_id=self.project_id,
      path=ProjectPaths.TopicEvaluation(key),
      wait=True,
    )
  def _load(self, key):
    paths = ProjectPathManager(project_id=self.project_id)
    file_path = paths.full_path(ProjectPaths.TopicEvaluation(key))
    FileNotExistsException.verify(file_path, error=FileNotExistsException.format_message(
      path=file_path,
      purpose="topic evaluation results",
      problem=f"It seems that you have not performed any evaluations on the topics of \"{key}\".",
    ))

    with self.lock(key):
      with open(file_path, 'r', encoding='utf-8') as f:
        try:
          result = TopicEvaluationResult.model_validate_json(f.read())
        except ValidationError:
          raise CorruptedFileException(
            CorruptedFileException.format_message(
              path=file_path,
              purpose="topic evaluation results",
            )
          )
      
    self.cache.set(CacheItem(
      key=key,
      value=result,
    ))
    return result
    
  def _save(self, value, key):
    paths = ProjectPathManager(project_id=self.project_id)
    file_path = paths.allocate_path(ProjectPaths.TopicEvaluation(key))
    with self.lock(key):
      with atomic_write(file_path, mode="text") as f:
        f.write(value.model_dump_json())


@dataclass
class BERTopicExperimentResultCacheAdapter(ProjectCacheAdapter[BERTopicExperimentResult]):
  def lock(self, key: str):
    return ProjectFileLockManager().lock_file(
      project_id=self.project_id,
      path=ProjectPaths.TopicModelExperiments(key),
      wait=True,
    )
  
  def _load(self, key):
    paths = ProjectPathManager(project_id=self.project_id)
    file_path = paths.full_path(ProjectPaths.TopicModelExperiments(key))
    FileNotExistsException.verify(
      file_path,
      error=FileNotExistsException.format_message(
        path=file_path,
        purpose="BERTopic experiment results",
        problem=f"It seems that you have not performed any BERTopic experiments on \"{key}\".",
      )
    )

    with self.lock(key):
      with open(file_path, 'r', encoding='utf-8') as f:
        try:
          result = BERTopicExperimentResult.model_validate_json(f.read())
        except ValidationError:
          raise CorruptedFileException(
            CorruptedFileException.format_message(
              path=file_path,
              purpose="BERTopic experiment results",
            )
          )
        
    self.cache.set(CacheItem(
      key=key,
      value=result,
    ))
    return result
    
  def _save(self, value, key):
    paths = ProjectPathManager(project_id=self.project_id)
    file_path = paths.allocate_path(ProjectPaths.TopicModelExperiments(key))
    with atomic_write(file_path, mode="text") as f:
      f.write(value.model_dump_json())
