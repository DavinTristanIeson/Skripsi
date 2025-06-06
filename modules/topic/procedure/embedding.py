
import numpy as np
import pandas as pd
from modules.project.paths import ProjectPaths
from modules.topic.bertopic_ext.builder import BERTopicIndividualModels

from .base import BERTopicProcedureComponent

from ..bertopic_ext import BERTopicEmbeddingModelPreprocessingPreference

class BERTopicEmbeddingProcedureComponent(BERTopicProcedureComponent):
  def run(self):
    # Dependencies
    column = self.state.column
    config = self.state.config
    mask = self.state.mask
    model = self.state.model
    cache = self.state.cache
    documents = self.state.documents
    embedding_documents = self.state.embedding_documents
    
    individual_models = BERTopicIndividualModels.cast(model)
    embedding_model = individual_models.embedding_model
    embedding_path = config.paths.full_path(ProjectPaths.DocumentEmbeddings(column.name))
    
    # Cache
    try:
      # First try to load cache as a dataframe
      embeddings = cache.document_vectors.load(column.name)
    except Exception:
      # If not, then don't even bother with cached stuff anymore
      embeddings = None

    if embeddings is not None:
      try:
        # Check if the mask can still be applied. If it still can, then everything's hunky-dory. File is still synced.
        cached_embeddings = embeddings.loc[mask, :]
        # Has NA values
        if cached_embeddings.isna().sum().sum() > 0:
          raise KeyError()
        # Else, just use the values
        self.state.document_vectors = cached_embeddings.to_numpy()
        self.task.log_success(f"Using cached document vectors for \"{column.name}\" from \"{embedding_path}\".")
        # no need to continue from here on out
        return
      except (KeyError, IndexError):
        # Otherwise, file is definitely not synced. Compute the new document vectors.
        self.task.log_error(f"Cached document vectors are no longer synced with the preprocessed documents ({len(embeddings)} vectors vs {len(documents)} documents). Recalculating the document vectors...")
        pass
     
    # Compute
    self.task.log_pending(f"Transforming documents of \"{column.name}\" into document vectors...")

    document_vector_mask = embedding_documents.notna()
    embedding_documents_masked = list(embedding_documents[document_vector_mask])
    embeddings = embedding_model.fit_transform(embedding_documents_masked) # type: ignore
    self.task.log_success(f"All documents from \"{column.name}\" has been successfully embedded using {column.topic_modeling.embedding_method}.")

    # Create a container first. Use mask.index as the source of truth for the indices.
    cached_document_vectors_raw = np.zeros((len(self.state.embedding_documents), embeddings.shape[1]), 
    dtype=np.float32)

    # Use DataFrame rather than npy file for safer .loc-based mapping to perform masking.
    # Numpy might not play well with pandas' boolean masks
    cached_document_vectors = pd.DataFrame(
      data=cached_document_vectors_raw,
      index=document_vector_mask.index,
      columns=list(map(str, range(embeddings.shape[1]))),
      dtype=pd.Float64Dtype(),
    )
    self.task.logger.debug(f"Shape of documents: {embedding_documents.shape}. Shape of mask: {document_vector_mask.shape} (True: {document_vector_mask.sum()}). Shape of embeddings: {embeddings.shape}. Shape of cached document vectors: {cached_document_vectors.shape}. Shape of embedding model input: {len(embedding_documents_masked)}")
    # Assign only the rows touched by mask; otherwise, leave them empty.
    cached_document_vectors.loc[document_vector_mask, :] = embeddings
    cached_document_vectors.loc[~document_vector_mask, :] = pd.NA
    # Save the document vectors
    cache.document_vectors.save(cached_document_vectors, column.name)
    
    # Effect
    self.state.document_vectors = cached_document_vectors.loc[mask, :].to_numpy()

    # self.task.logger.debug(str(cached_document_vectors.loc[mask, :]))
    # self.task.logger.debug(f"NA values in document vectors: {cached_document_vectors.loc[mask, :].isna().sum().sum()}. Document vectors type: {self.state.document_vectors.dtype}. Intersection: {(mask & document_vector_mask).sum()} (Mask: {mask.sum()}, Document Vector Mask: {document_vector_mask.sum()})") # type: ignore

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