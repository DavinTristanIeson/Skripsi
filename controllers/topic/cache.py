import abc
import os
import numpy as np
import numpy.typing as npt

from models.config.paths import ProjectPathManager

class UnavailableCacheException(Exception):
  pass

class CachedEmbeddingModel(abc.ABC):
  @property
  @abc.abstractmethod
  def embedding_path(self)->str:
    pass

  def save_embeddings(self, embeddings: npt.NDArray):
    os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)
    np.save(self.embedding_path, embeddings)

  def has_cached_embeddings(self):
    return os.path.exists(self.embedding_path)
  
  def load_cached_embeddings(self)->npt.NDArray:
    if self.has_cached_embeddings():
      return np.load(self.embedding_path)
    raise UnavailableCacheException(f"There are no cached embeddings in {self.embedding_path}. This is a developer oversight; remember to add a self.has_cached_embeddings check before calling this function.")
