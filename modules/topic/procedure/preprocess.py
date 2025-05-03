from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional, Sequence

from modules.api.wrapper import ApiError
from modules.project.cache import ProjectCacheManager
from modules.project.paths import ProjectPaths
from modules.topic.bertopic_ext.builder import BERTopicModelBuilder
from modules.topic.procedure.base import BERTopicProcedureComponent


class BERTopicDataLoaderProcedureComponent(BERTopicProcedureComponent):
  def run(self):
    # Dependencies
    config = self.state.config

    # Compute
    workspace_path = config.paths.full_path(ProjectPaths.Workspace)
    self.task.log_pending(f"Loading cached dataset from \"{workspace_path}\"...")
    self.state.cache.load_workspace() # Load workspace in cache
    self.task.log_success(f"Loaded cached dataset from \"{workspace_path}\"...")


@dataclass
class BERTopicPreprocessProcedureComponent(BERTopicProcedureComponent):
  can_save: bool = True
  def run(self):
    # Dependencies
    column = self.state.column
    cache = self.state.cache
    df = cache.load_workspace()
    config = self.state.config
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
      cache.save_workspace(df)
    
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
    df = cache.load_workspace()
    config = self.state.config
    preprocess_name = column.preprocess_column.name

    raw_documents = df[column.name]
    if column.preprocess_column.name not in df.columns:
      raise ApiError(f"There are no cached preprocessed documents. Please run the topic modeling algorithm first.", HTTPStatus.UNPROCESSABLE_ENTITY)
    
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