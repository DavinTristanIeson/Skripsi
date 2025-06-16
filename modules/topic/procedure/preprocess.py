from dataclasses import dataclass
import threading
from typing import Sequence, cast

import numpy as np
import pandas as pd

from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.project.cache import ProjectCache
from modules.project.paths import ProjectPathManager
from modules.topic.bertopic_ext.embedding import BERTopicEmbeddingModelPreprocessingPreference, get_embedding_model_preference
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

    raw_documents_series = df[column.name]
    document_vector_mask = raw_documents_series.notna() & (raw_documents_series.str.len() > 0)
    if column.preprocess_column.name in df.columns:
      # Cache
      raw_preprocess_documents = df[preprocess_name]
      mask = df[preprocess_name].notna()
      preprocess_documents = raw_preprocess_documents[mask]
      self.task.log_success(f"Using the preprocessed documents in column \"{column.preprocess_column.name}\".")
    else:
      # Compute
      self.task.log_pending(f"Preprocessing the documents in column \"{column.name}\". Text preprocessing may take some time...")
      # preprocess_topic_keywords set NA for invalid documents, so we need to recompute mask
      df.loc[document_vector_mask, preprocess_name] = column.preprocessing.preprocess_heavy(
        cast(Sequence[str], raw_documents_series[document_vector_mask])
      )
      df.loc[~document_vector_mask, preprocess_name] = pd.NA
      mask = df[preprocess_name].notna()
      preprocess_documents = df.loc[mask, preprocess_name]
      self.task.log_success(f"Finished preprocessing the documents in column \"{column.name}\".")
      cache.workspaces.save(df)
    
    if len(preprocess_documents) == 0:
      raise ValueError(f"\"{column.name}\" does not contain any valid documents after the preprocessing step. Either change the preprocessing configuration of \"{column.name}\" to be more lax (e.g: lower the min word frequency, min document length), or set the type of this column to Unique.")
    
    try:
      document_vectors = cache.document_vectors.load(column.name)
      # Try to access mask. If this fails, then documents are desynced with cached embeddings.
      if np.any(np.isnan(document_vectors[mask, :])):
        self.task.logger.warning(f"Cached document vectors contains NA values, which indicates that they are desynced with the current documents. Document vectors will need to be recalculated.")
        document_vectors = None
    except Exception as e:
      self.task.logger.debug(f"An exception occurred while loading document vectors: {e}. Documents will be lightly preprocessed just in case the embedding model needs them.")
      document_vectors = None

    if document_vectors is None:
      # Only preprocess lightly if document_vectors doesn't exist
      self.task.log_pending(f"Performing light preprocessing for the documents in column \"{column.name}\". This shouldn't take too long...")
      # Light preprocessing for SBERT
      # Preprocess everything at once, including documents that are not included in the documents after preprocessing. This is so that we can make better use of the document vector cache.
      # Otherwise document vector cache has to be recalculated whenever document preprocessing accidentally excludes some previously existing document.
      embedding_documents = raw_documents_series.copy()
      embedding_documents[document_vector_mask] = column.preprocessing.preprocess_light(
        cast(Sequence[str], raw_documents_series[document_vector_mask])
      )
      embedding_documents[~document_vector_mask] = pd.NA
      self.task.log_success(f"Finished performing light preprocessing for the documents in column \"{column.name}\".")
    else:
      embedding_documents = raw_documents_series.copy()
      self.task.log_success(f"As there are cached document vectors, light preprocessing will not be performed for the documents in column \"{column.name}\".")
    # Else, no need to perform any more light preprocessing

    self.task.logger.debug(f"[Preprocessing] Documents count: {len(preprocess_documents)}")
    # Effect
    self.state.mask = mask
    if get_embedding_model_preference(column) == BERTopicEmbeddingModelPreprocessingPreference.Heavy:
      self.state.embedding_documents = preprocess_documents
    else:
      self.state.embedding_documents = raw_documents_series
    self.state.documents = preprocess_documents # type: ignore

@dataclass
class BERTopicCacheOnlyPreprocessProcedureComponent(BERTopicProcedureComponent):
  def run(self):
    # Dependencies
    column = self.state.column
    cache = self.state.cache
    df = cache.workspaces.load(cached=False)
    preprocess_name = column.preprocess_column.name

    column.assert_internal_columns(df, with_preprocess=True, with_topics=False)
    
    # Cache
    raw_preprocess_documents = df[preprocess_name]
    mask = df[preprocess_name].notna()
    preprocess_documents = raw_preprocess_documents[mask]
    
    # Effect
    self.state.mask = mask
    self.state.documents = preprocess_documents # type: ignore

__all__ = [
  "BERTopicDataLoaderProcedureComponent",
  "BERTopicPreprocessProcedureComponent"
]