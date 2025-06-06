
from modules.topic.bertopic_ext.builder import BERTopicIndividualModels

from .base import BERTopicProcedureComponent

from ..bertopic_ext import BERTopicEmbeddingModelPreprocessingPreference

class BERTopicEmbeddingProcedureComponent(BERTopicProcedureComponent):
  def run(self):
    # Dependencies
    column = self.state.column
    model = self.state.model
    documents = self.state.documents
    
    individual_models = BERTopicIndividualModels.cast(model)
    embedding_model = individual_models.embedding_model
    
    # Cache
    embeddings = embedding_model.load_cached_embeddings()
    if embeddings is not None:
      self.task.logger.debug(f"[Embedding] Cached document vectors shape: {embeddings.shape}, versus documents count: {len(documents)}")
      if len(embeddings) == len(documents):
        self.state.document_vectors = embeddings
        self.task.log_success(f"Using cached document vectors for \"{column.name}\" from \"{embedding_model.embedding_path}\".")
        return
      else:
        self.task.log_error(f"Cached document vectors are no longer synced with the preprocessed documents ({len(embeddings)} vectors vs {len(documents)} documents). Recalculating the document vectors...")
     
    # Compute
    self.task.log_pending(f"Transforming documents of \"{column.name}\" into document vectors...")

    if embedding_model.preference() == BERTopicEmbeddingModelPreprocessingPreference.Light:
      input_documents = self.state.embedding_documents
    else:
      input_documents = self.state.documents

    embeddings = embedding_model.fit_transform(input_documents) # type: ignore
    self.task.logger.debug(f"[Embedding] Document vectors shape: {embeddings.shape}, versus documents count: {len(input_documents)}")
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