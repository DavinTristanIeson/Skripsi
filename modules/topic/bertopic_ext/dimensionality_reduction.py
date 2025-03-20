from dataclasses import dataclass, field
import functools
import http
from typing import TYPE_CHECKING, cast
import abc

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from modules.api import ApiError
from modules.project.paths import ProjectPathManager, ProjectPaths
from modules.config import TextualSchemaColumn
from modules.storage import CachedEmbeddingTransformerBehavior

if TYPE_CHECKING:
  from umap import UMAP
@dataclass
class __CachedUMAP(CachedEmbeddingTransformerBehavior, abc.ABC, BaseEstimator, TransformerMixin):
  project_id: str
  column: TextualSchemaColumn
  low_memory: bool
  
  @property
  @abc.abstractmethod
  def model(self)->"UMAP":
    ...

  def _fit(self, X: np.ndarray):
    self.model.fit(X)
  
  def _transform(self, X: np.ndarray):
    return cast(np.ndarray, self.model.transform(X))
  
  @property
  def embedding_path(self):
    paths = ProjectPathManager(project_id=self.project_id)
    return paths.full_path(ProjectPaths.UMAPEmbeddings(self.column.name))

class BERTopicCachedUMAP(__CachedUMAP):
  @functools.cached_property
  def __model(self):
    from umap import UMAP
    return UMAP(
      n_neighbors=self.column.topic_modeling.reference_document_count
        or self.column.topic_modeling.min_topic_size,
      min_dist=0.1,
      # BERTopic uses 5 dimensions
      n_components=5,
      metric="euclidean",
      low_memory=self.low_memory
    )
  
  @property
  def model(self):
    return self.__model


@dataclass
class VisualizationCachedUMAPResult:
  document_embeddings: np.ndarray
  topic_embeddings: np.ndarray

@dataclass
class VisualizationCachedUMAP(__CachedUMAP):
  corpus_size: int
  topic_count: int
  @property
  def embedding_path(self):
    paths = ProjectPathManager(project_id=self.project_id)
    return paths.full_path(ProjectPaths.VisualizationEmbeddings(self.column.name))

  @functools.cached_property
  def __model(self):
    from umap import UMAP
    return UMAP(
      n_neighbors=self.column.topic_modeling.reference_document_count
        or self.column.topic_modeling.min_topic_size,
      min_dist=0.1,
      n_components=2,
      metric="euclidean",
      low_memory=self.low_memory
    )
  
  @property
  def model(self):
    return self.__model
  
  def join_embeddings(self, document_embeddings: np.ndarray, topic_embeddings: np.ndarray):
    return np.vstack([document_embeddings, topic_embeddings])
  
  def separate_embeddings(self, embeddings: np.ndarray):
    expected_length = self.corpus_size + self.topic_count
    if embeddings.shape[0] != expected_length:
      raise ApiError(f"Expected cached visualization embeddings for \"{self.column.name}\" to have {self.corpus_size} + {self.topic_count} rows, but got {embeddings.shape[0]} instead. Maybe the cached visualization embeddings in \"{self.embedding_path}\" has been corrupted. To fix this, please run the topic modeling procedure again.", http.HTTPStatus.INTERNAL_SERVER_ERROR)

    return VisualizationCachedUMAPResult(
      document_embeddings=embeddings[:self.corpus_size],
      topic_embeddings=embeddings[self.corpus_size:],
    )
  
__all__ = [
  "BERTopicCachedUMAP",
  "VisualizationCachedUMAPResult",
  "VisualizationCachedUMAP",
]