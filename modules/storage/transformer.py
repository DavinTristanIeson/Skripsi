import abc
from dataclasses import dataclass, field
import os
from typing import Any, Generic, Optional, TypeVar, cast

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from modules.logger import ProvisionedLogger
from modules.logger.time import TimeLogger

logger = ProvisionedLogger().provision("Cached Tranformer Pipeline")

TModel = TypeVar("TModel")
TInput = TypeVar("TInput")


class CachedEmbeddingBehavior(abc.ABC):
  @property
  @abc.abstractmethod
  def embedding_path(self)->str:
    pass

  def save_embeddings(self, embeddings: np.ndarray):
    logger.info(f"Saving embeddings with shape {embeddings.shape} to \"{self.embedding_path}\".")
    try:
      os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)
      np.save(self.embedding_path, embeddings)
      logger.info(f"Saved embeddings to \"{self.embedding_path}\".")
    except Exception as e:
      # Silent failure. Continue process as normal.
      logger.error(f"Failed to save embeddings to \"{self.embedding_path}\".")
      logger.error(e)

  def load_cached_embeddings(self)->Optional[np.ndarray]:
    if not os.path.exists(self.embedding_path):
      return None
    try:
      embeddings = np.load(self.embedding_path)
      logger.info(f"Loaded cached embeddings from \"{self.embedding_path}\" with shape {embeddings.shape}.")
      return embeddings
    except Exception as e:
      # Silent failure. Continue process as normal.
      logger.exception(e)
      logger.info(f"Failed to load cached embeddings from \"{self.embedding_path}\".")
      return None
    
  def delete_cached_embeddings(self):
    if not os.path.exists(self.embedding_path):
      logger.warning(f"Requested deletion of \"{self.embedding_path}\", but there's no file in that location.")
      return
    try:
      os.remove(self.embedding_path)
      logger.info(f"Deleted cached embeddings from \"{self.embedding_path}\".")
      return
    except Exception as e:
      # Silent failure. Continue process as normal.
      logger.exception(e)
      logger.info(f"Failed to delete cached embeddings from \"{self.embedding_path}\".")
      return

@dataclass
class CachedEmbeddingTransformerBehavior(CachedEmbeddingBehavior, abc.ABC, Generic[TModel, TInput]):
  model: TModel = field(init=False)

  @abc.abstractmethod
  def _transform(self, X: TInput)->np.ndarray:
    ...

  def transform(self, X: TInput):
    cached_embeddings = self.load_cached_embeddings()
    if cached_embeddings is not None:
      if not getattr(X, "__len__"):
        logger.debug(f"Embedding transformer input doesn't support __len__. We cannot check if shape is synced.")
        return cached_embeddings
      
      X_length = len(X) # type: ignore
      logger.debug(f"Comparing shape of cached embeddings (shape: {cached_embeddings.shape}) with length of embedding transformer input (length: {X_length})")
      if X_length == len(cached_embeddings): # type: ignore
        logger.debug(f"Shape of cached embeddings matches input shape. Cached embeddings will be reused.")
        return cached_embeddings
      else:
        logger.debug(f"Shape of cached embeddings doesn't match input shape. Embeddings will be recalculated.")
    
    embeddings = self._transform(X)
    self.save_embeddings(embeddings)
    return embeddings
  
__all__ = [
  "CachedEmbeddingBehavior",
  "CachedEmbeddingTransformerBehavior",
]