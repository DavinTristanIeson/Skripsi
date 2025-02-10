from dataclasses import dataclass
import functools
from typing import TYPE_CHECKING
import abc

from controllers.topic.cache import CachedEmbeddingModel
from models.config import ProjectPathManager, TextualSchemaColumn, ProjectPaths
import numpy.typing as npt
if TYPE_CHECKING:
  from sklearn.base import BaseEstimator, TransformerMixin
  from umap import UMAP

@dataclass
class CachedUMAP(CachedEmbeddingModel, abc.ABC, BaseEstimator, TransformerMixin):
  paths: ProjectPathManager
  column: TextualSchemaColumn
  
  @property
  @abc.abstractmethod
  def model(self)->UMAP:
    ...

  def fit(self, X: npt.NDArray):
    if self.has_cached_embeddings():
      return self
    self.model.fit(X)
    return self
  
  def transform(self, X: npt.NDArray):
    if self.has_cached_embeddings():
      return self.load_cached_embeddings()
    return self.model.transform(X)

class BERTopicCachedUMAP(CachedUMAP):
  @property
  def embedding_path(self):
    return self.paths.full_path(ProjectPaths.UMAPEmbeddings(self.column.name))

  @functools.cached_property
  def __model(self):
    from umap import UMAP
    return UMAP(
      n_neighbors=self.column.topic_modeling.globality_consideration
        or self.column.topic_modeling.min_topic_size,
      min_dist=0.1,
      # BERTopic uses 5 dimensions
      n_components=5,
      metric="euclidean",
      low_memory=self.column.topic_modeling.low_memory
    )
  
  @property
  def model(self):
    return self.__model

class VisualizationCachedUMAP(CachedUMAP):
  @property
  def embedding_path(self):
    return self.paths.full_path(ProjectPaths.VisualizationEmbeddings(self.column.name))

  @functools.cached_property
  def __model(self):
    from umap import UMAP
    return UMAP(
      n_neighbors=self.column.topic_modeling.globality_consideration
        or self.column.topic_modeling.min_topic_size,
      min_dist=0.1,
      n_components=2,
      metric="euclidean",
      low_memory=self.column.topic_modeling.low_memory
    )
  
  @property
  def model(self):
    return self.__model