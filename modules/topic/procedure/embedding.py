
from http import HTTPStatus
from modules.api.wrapper import ApiError
from modules.topic.bertopic_ext.builder import BERTopicIndividualModels

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
    column = self.state.column
    model = self.state.model
    
    individual_models = BERTopicIndividualModels.cast(model)
    embedding_model = individual_models.embedding_model
    
    # Cache
    embeddings = embedding_model.load_cached_embeddings()
    if embeddings is None:
      raise ApiError(f"There are no cached embeddings in \"{embedding_model.embedding_path}\". Please run the topic modeling algorithm first.", HTTPStatus.UNPROCESSABLE_ENTITY)
    
    self.task.log_success(f"Using cached document vectors for \"{column.name}\" from \"{embedding_model.embedding_path}\".")
    self.state.document_vectors = embeddings
    
    # Effect
    self.state.document_vectors = embeddings


__all__ = [
  "BERTopicEmbeddingProcedureComponent",
]