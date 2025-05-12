from dataclasses import dataclass
import threading
from typing import Sequence, cast

from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.project.cache import ProjectCache
from modules.project.paths import ProjectPathManager
from modules.topic.procedure.base import BERTopicProcedureComponent

@dataclass
class BERTopicDataLoaderProcedureComponent(BERTopicProcedureComponent):
  project_id: str
  column: str
  def run(self):
    # Compute
    base_path = ProjectPathManager(project_id=self.project_id).base_path
    self.task.log_pending(f"Loading configuration from project in \"{base_path}\"...")

    cache = ProjectCache(
      project_id=self.project_id,
      lock=threading.RLock(),
    )
    config = cache.config
    column = config.data_schema.assert_of_type(self.column, [SchemaColumnTypeEnum.Textual])

    self.task.log_success(f"Successfully loaded configuration from project in \"{base_path}\"...")

    # Effects
    self.state.config = config
    self.state.column = cast(TextualSchemaColumn, column)
    self.state.cache = cache


@dataclass
class BERTopicPreprocessProcedureComponent(BERTopicProcedureComponent):
  can_save: bool = True
  def run(self):
    # Dependencies
    column = self.state.column
    cache = self.state.cache
    df = cache.workspaces.load(cached=False)
    preprocess_name = column.preprocess_column.name

    raw_documents = df[column.name]
    if column.preprocess_column.name in df.columns:
      # Cache
      raw_preprocess_documents = df[preprocess_name]
      mask = df[preprocess_name].notna()
      preprocess_documents = raw_preprocess_documents[mask]
    else:
      # Compute
      original_mask = raw_documents.notna()
      original_documents: Sequence[str] = raw_documents[original_mask] # type: ignore

      self.task.log_pending(f"Preprocessing the documents in column \"{column.name}\". Text preprocessing may take some time...")
      # preprocess_topic_keywords set NA for invalid documents, so we need to recompute mask
      df.loc[original_mask, preprocess_name] = column.preprocessing.preprocess_heavy(original_documents) # type: ignore
      mask = df[preprocess_name].notna()
      preprocess_documents = df.loc[mask, preprocess_name]
      self.task.log_success(f"Finished preprocessing the documents in column \"{column.name}\".")
      cache.workspaces.save(df)
    
    if len(preprocess_documents) == 0:
      raise ValueError(f"\"{column.name}\" does not contain any valid documents after the preprocessing step. Either change the preprocessing configuration of \"{column.name}\" to be more lax (e.g: lower the min word frequency, min document length), or set the type of this column to Unique.")
    
    original_documents: Sequence[str] = raw_documents[mask] # type: ignore
    self.task.log_pending(f"Performing light preprocessing for the documents in column \"{column.name}\". This shouldn't take too long...")
    # Light preprocessing for SBERT
    sbert_documents = column.preprocessing.preprocess_light(original_documents)
    self.task.log_success(f"Finished performing light preprocessing for the documents in column \"{column.name}\". {len(original_documents) - len(preprocess_documents)} document(s) has been excluded from the topic modeling process.")

    # Effect
    self.state.mask = mask
    self.state.embedding_documents = sbert_documents
    self.state.documents = preprocess_documents # type: ignore

@dataclass
class BERTopicCacheOnlyPreprocessProcedureComponent(BERTopicProcedureComponent):
  def run(self):
    # Dependencies
    column = self.state.column
    cache = self.state.cache
    df = cache.workspaces.load(cached=False)
    preprocess_name = column.preprocess_column.name

    raw_documents = df[column.name]
    column.assert_internal_columns(df, with_preprocess=True, with_topics=False)
    
    # Cache
    raw_preprocess_documents = df[preprocess_name]
    mask = df[preprocess_name].notna()
    preprocess_documents = raw_preprocess_documents[mask]
    
    original_documents: Sequence[str] = raw_documents[mask] # type: ignore
    self.task.log_pending(f"Performing light preprocessing for the documents in column \"{column.name}\". This shouldn't take too long...")
    sbert_documents = column.preprocessing.preprocess_light(original_documents)
    # Light preprocessing for SBERT
    self.task.log_success(f"Finished performing light preprocessing for the documents in column \"{column.name}\". {len(original_documents) - len(preprocess_documents)} document(s) has been excluded from the topic modeling process.")

    # Effect
    self.state.mask = mask
    self.state.embedding_documents = sbert_documents
    self.state.documents = preprocess_documents # type: ignore

__all__ = [
  "BERTopicDataLoaderProcedureComponent",
  "BERTopicPreprocessProcedureComponent"
]