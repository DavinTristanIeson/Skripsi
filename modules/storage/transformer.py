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
    logger.info(f"Saving embeddings to \"{self.embedding_path}\".")
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
      logger.info(f"Loaded cached embeddings from \"{self.embedding_path}\".")
      return embeddings
    except Exception as e:
      # Silent failure. Continue process as normal.
      logger.exception(e)
      logger.info(f"Failed to load cached embeddings from \"{self.embedding_path}\".")
      return None

class SavedModelBehavior(abc.ABC, Generic[TModel]):
  @property
  @abc.abstractmethod
  def embedding_model_path(self)->str:
    ...
  
  @abc.abstractmethod
  def _save_model(self, model: TModel):
    ...

  @abc.abstractmethod
  def _load_model(self)->TModel:
    ...

  @abc.abstractmethod
  def load_default_model(self)->TModel:
    ...

  def save_model(self, model: TModel):
    os.makedirs(os.path.dirname(self.embedding_model_path), exist_ok=True)
    self._save_model(model)
    logger.info(f"Saved model in \"{self.embedding_model_path}\"")

  def load_model(self)->Optional[TModel]:
    if not os.path.exists(self.embedding_model_path):
      return None
    try:
      model = cast(TModel, self._load_model())
      return model
    except Exception as e:
      logger.exception(e)
      logger.error(f"Failed to load cached model from \"{self.embedding_model_path}\". Creating a new model...")
      return None

@dataclass
class SavedModelTransformerBehavior(SavedModelBehavior, abc.ABC, Generic[TModel, TInput]):
  model: TModel = field(init=False)

  @abc.abstractmethod
  def _fit(self, X: TInput):
    ...

  def fit(self, X: TInput):
    model = self.load_model()
    if model:
      self.model = model
      return self
    model = self.load_default_model()
    self.model = model

    self._fit(X)

    self.save_model(model)
    return self

@dataclass
class CachedEmbeddingTransformerBehavior(CachedEmbeddingBehavior, abc.ABC, Generic[TModel, TInput]):
  model: TModel = field(init=False)

  @abc.abstractmethod
  def _transform(self, X: TInput)->np.ndarray:
    ...

  def transform(self, X: TInput):
    cached_embeddings = self.load_cached_embeddings()
    if cached_embeddings:
      return cached_embeddings
    
    embeddings = self._transform(X)
    self.save_embeddings(embeddings)
    return embeddings
  
__all__ = [
  "CachedEmbeddingBehavior",
  "SavedModelBehavior",
  "CachedEmbeddingTransformerBehavior",
  "SavedModelTransformerBehavior"
]