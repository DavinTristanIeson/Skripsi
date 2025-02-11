import abc
import os
from typing import Optional
import numpy as np

from common.logger import RegisteredLogger
from models.config.paths import ProjectPathManager

logger = RegisteredLogger().provision("Topic Modeling")

class CachedEmbeddingModel(abc.ABC):
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
      logger.info(f"Failed to load cached embeddings from \"{self.embedding_path}\".")
      logger.error(e)
      return None