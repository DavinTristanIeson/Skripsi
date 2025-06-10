from dataclasses import dataclass
import functools
from typing import TYPE_CHECKING, cast
import abc

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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

  def fit(self, X: np.ndarray):
    return self

  def _transform(self, X: np.ndarray):
    low_dimensional_points = self.model.fit_transform(X)
    return cast(np.ndarray, low_dimensional_points)


class BERTopicCachedUMAP(__CachedUMAP):
  @functools.cached_property
  def __model(self):
    from umap import UMAP
    return UMAP(
      random_state=2025,
      n_neighbors=self.column.topic_modeling.reference_document_count,
      min_dist=0.1,
      n_jobs=1,
      # BERTopic uses 5 dimensions
      n_components=5,
      metric="cosine",
      low_memory=self.low_memory
    )
  
  @property
  def model(self):
    return self.__model

  @property
  def embedding_path(self):
    paths = ProjectPathManager(project_id=self.project_id)
    return paths.full_path(ProjectPaths.UMAPEmbeddings(self.column.name))

@dataclass
class VisualizationCachedUMAP(__CachedUMAP):
  @property
  def embedding_path(self):
    paths = ProjectPathManager(project_id=self.project_id)
    return paths.full_path(ProjectPaths.VisualizationEmbeddings(self.column.name))

  @functools.cached_property
  def __model(self):
    from umap import UMAP
    return UMAP(
      random_state=2025,
      n_neighbors=10, # BERTopic default
      min_dist=0.1,
      n_components=2,
      n_jobs=1,
      # This performs dimensionality reduction on UMAP vectors, so we use euclidean metric rather than cosine distance.
      metric="euclidean",
      low_memory=self.low_memory
    )
  
  @property
  def model(self):
    return self.__model

__all__ = [
  "BERTopicCachedUMAP",
  "VisualizationCachedUMAP",
]