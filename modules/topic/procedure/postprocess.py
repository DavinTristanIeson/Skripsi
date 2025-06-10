from dataclasses import dataclass
import numpy as np
import pandas as pd

from modules.logger.provisioner import ProvisionedLogger
from modules.topic.bertopic_ext.dimensionality_reduction import BERTopicCachedUMAP
from modules.topic.exceptions import FoundNoTopicsException
from modules.topic.procedure.base import BERTopicProcedureComponent


from ..bertopic_ext import (
  BERTopicInterpreter,
  VisualizationCachedUMAP,
)

from ..model import TopicModelingResult

logger = ProvisionedLogger().provision("Topic Modeling")
class BERTopicVisualizationEmbeddingProcedureComponent(BERTopicProcedureComponent):
  def run(self):
    # Dependencies
    config = self.state.config
    column = self.state.column
    cache = self.state.cache
    
    umap_model = BERTopicCachedUMAP(
      column=column,
      project_id=config.project_id,
      low_memory=True,
    )

    # Compute
    self.task.log_pending("Mapping the document vectors to 2D for visualization purposes...")
    vis_umap_model = VisualizationCachedUMAP(
      project_id=config.project_id,
      column=column,
      low_memory=True,
    )
    document_vectors = umap_model.load_cached_embeddings()
    if document_vectors is None:
      self.task.log_error("Failed to reuse the reduced document vectors calculated by UMAP. Perhaps this is a developer oversight. UMAP will be executed again on the original document vectors.")
      document_vectors = self.state.document_vectors
    
    vis_umap_model.fit_transform(document_vectors)
    cache.visualization_vectors.invalidate()

    self.task.log_success(f"Finished mapping the document vectors to 2D. The visualization vectors have been stored in {vis_umap_model.embedding_path}.")

@dataclass
class BERTopicPostprocessProcedureComponent(BERTopicProcedureComponent):
  can_save: bool = True
  def run(self):
    # Dependencies
    column = self.state.column
    config = self.state.config
    cache = self.state.cache
    documents = self.state.documents
    model = self.state.model
    df = cache.workspaces.load(cached=False)
    mask = self.state.mask

    self.task.log_pending(f"Applying post-processing on the topics of \"{column.name}\"...")

    # Set topic assignments
    document_topic_mapping_column = pd.Series(np.full(len(mask), -1), index=mask.index, dtype="Int32")
    document_topic_mapping_column.loc[mask] = model.topics_ # type: ignore
    document_topic_mapping_column.loc[~mask] = pd.NA # type: ignore
    df[column.topic_column.name] = document_topic_mapping_column

    topics = BERTopicInterpreter(model).extract_topics()
    if len(topics) == 0:
      raise FoundNoTopicsException(column=column.name)

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
    self.state.result = topic_modeling_result
    if self.can_save:
      cache = self.state.cache
      cache.workspaces.save(df)
      cache.topics.save(topic_modeling_result, column.name)


__all__ = [
  "BERTopicPostprocessProcedureComponent",
  "BERTopicVisualizationEmbeddingProcedureComponent",
]
  