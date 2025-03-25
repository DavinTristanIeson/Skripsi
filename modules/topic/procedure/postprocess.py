from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCacheManager
from modules.topic.bertopic_ext.builder import BERTopicIndividualModels
from modules.topic.bertopic_ext.dimensionality_reduction import BERTopicCachedUMAP
from modules.topic.procedure.base import BERTopicProcedureComponent


from ..bertopic_ext import (
  BERTopicInterpreter,
  VisualizationCachedUMAP,
  VisualizationCachedUMAPResult
)

from ..model import TopicModelingResult
if TYPE_CHECKING:
  from bertopic import BERTopic

logger = ProvisionedLogger().provision("Topic Modeling")
class BERTopicVisualizationEmbeddingProcedureComponent(BERTopicProcedureComponent):
  def run(self):
    # Dependencies
    config = self.state.config
    column = self.state.column
    model = self.state.model
    documents = self.state.documents
    document_topic_assignments = np.array(model.topics_)
    interpreter = BERTopicInterpreter(self.state.model)
    umap_model = BERTopicCachedUMAP(
      column=column,
      project_id=config.project_id,
      low_memory=True,
    )

    # Compute
    self.task.log_pending("Mapping the document and topic vectors to 2D for visualization purposes...")
    vis_umap_model = VisualizationCachedUMAP(
      project_id=config.project_id,
      column=column,
      topic_count=interpreter.topic_count,
      corpus_size=len(documents),
      low_memory=True,
    )
    document_vectors = umap_model.load_cached_embeddings()
    if document_vectors is None:
      self.task.log_error("Failed to reuse the reduced document vectors calculated by UMAP. Perhaps this is a developer oversight. UMAP will be executed again on the original document vectors.")
      document_vectors = self.state.document_vectors
    
    logger.debug(f"Calculating topic vectors from document vectors with the following configuration: {document_topic_assignments}")
    raw_topic_vectors: list[np.ndarray] = []
    for topic in range(interpreter.topic_count):
      mean_document_vector = document_vectors[document_topic_assignments == topic].mean(axis=0)
      raw_topic_vectors.append(mean_document_vector)
    topic_vectors = np.array(raw_topic_vectors)
    if topic_vectors.shape[0] == 0:
      topic_vectors = np.full((interpreter.topic_count, document_vectors.shape[1]), 0, dtype=document_vectors.dtype)
    logger.debug(f"Concatening document vectors with shape {document_vectors.shape} and topic vectors with shape {topic_vectors.shape}.")

    high_dimensional_vectors = np.vstack([document_vectors, topic_vectors])
    vis_umap_model.fit_transform(high_dimensional_vectors)

    self.task.log_success(f"Finished mapping the document and topic vectors to 2D. The embeddings have been stored in {vis_umap_model.embedding_path}.")

class BERTopicPostprocessProcedureComponent(BERTopicProcedureComponent):
  def run(self):
    # Dependencies
    column = self.state.column
    config = self.state.config
    documents = self.state.documents
    model = self.state.model
    df = self.state.df
    mask = self.state.mask

    self.task.log_pending(f"Applying post-processing on the topics of \"{column.name}\"...")

    # Set topic assignments
    document_topic_mapping_column = pd.Series(np.full(len(df), -1), dtype=np.int32)
    document_topic_mapping_column[mask] = model.topics_
    document_topic_mapping_column[~mask] = pd.NA
    df[column.topic_column.name] = document_topic_mapping_column

    topics = BERTopicInterpreter(model).extract_topics()

    topic_modeling_result = TopicModelingResult(
      project_id=config.project_id,
      topics=topics,
      valid_count=len(documents),
      total_count=len(mask),
      invalid_count=len(mask) - mask.sum(),
      outlier_count=(document_topic_mapping_column == -1).sum()
    )
    self.task.log_success(f"Finished appling post-processing on the topics of \"{column.name}\".")

    # Effect
    cache = ProjectCacheManager().get(config.project_id)
    cache.save_workspace(df)
    cache.save_topic(topic_modeling_result, column.name)
    ProjectCacheManager().invalidate(config.project_id)

    self.state.result = topic_modeling_result

__all__ = [
  "BERTopicPostprocessProcedureComponent",
  "BERTopicVisualizationEmbeddingProcedureComponent",
]
  