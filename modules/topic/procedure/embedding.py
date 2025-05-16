
from http import HTTPStatus
from modules.api.wrapper import ApiError
from modules.topic.bertopic_ext.builder import BERTopicIndividualModels
from modules.topic.bertopic_ext.dimensionality_reduction import BERTopicCachedUMAP
from modules.topic.exceptions import MissingCachedTopicModelingResult, RequiresTopicModelingException, UnsyncedDocumentVectorsException

from .base import BERTopicProcedureComponent

from ..bertopic_ext import BERTopicEmbeddingModelPreprocessingPreference

class BERTopicEmbeddingProcedureComponent(BERTopicProcedureComponent):
  def run(self):
    # Dependencies
    column = self.state.column
    model = self.state.model
    
    individual_models = BERTopicIndividualModels.cast(model)
    embedding_model = individual_models.embedding_model
    
    # Cache
    embeddings = embedding_model.load_cached_embeddings()
    if embeddings is not None:
      self.task.log_success(f"Using cached document vectors for \"{column.name}\" from \"{embedding_model.embedding_path}\".")
      self.state.document_vectors = embeddings
      return
    
    # Compute
    self.task.log_pending(f"Transforming documents of \"{column.name}\" into document vectors...")

    if embedding_model.preference() == BERTopicEmbeddingModelPreprocessingPreference.Light:
      input_documents = self.state.embedding_documents
    else:
      input_documents = self.state.documents

    embeddings = embedding_model.fit_transform(input_documents) # type: ignore
    self.task.log_success(f"All documents from \"{column.name}\" has been successfully embedded using {column.topic_modeling.embedding_method} and saved in \"{embedding_model.embedding_path}\".")
    
    # Effect
    self.state.document_vectors = embeddings

class BERTopicCacheOnlyEmbeddingProcedureComponent(BERTopicProcedureComponent):
  def run(self):
    # Dependencies
    cache = self.state.cache
    column = self.state.column

    # Compute
    cached_umap_vectors = cache.umap_vectors.load(column.name)

    # Effect
    self.state.document_vectors = cached_umap_vectors


__all__ = [
  "BERTopicEmbeddingProcedureComponent",
]