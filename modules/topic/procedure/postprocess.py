from dataclasses import dataclass
import functools
import http
import itertools
from typing import TYPE_CHECKING, Sequence, cast
import numpy as np
import pandas as pd
import pydantic

from modules.api import ApiError
from modules.topic.procedure.hierarchy import bertopic_topic_hierarchy

from ..bertopic_ext import (
  BERTopicInterpreter,
  VisualizationCachedUMAP,
  VisualizationCachedUMAPResult
)

from .utils import _BERTopicColumnIntermediateResult
from ..model import TopicModelingResult, Topic
if TYPE_CHECKING:
  from bertopic import BERTopic


def bertopic_visualization_embeddings(
  intermediate: _BERTopicColumnIntermediateResult
)->VisualizationCachedUMAPResult:
  task = intermediate.task
  config = intermediate.config
  column = intermediate.column
  model = intermediate.model
  documents = intermediate.documents
  embeddings = intermediate.embeddings    

  interpreter = BERTopicInterpreter(intermediate.model)

  task.log_pending("Mapping the document and topic vectors to 2D for visualization purposes...")
  vis_umap_model = VisualizationCachedUMAP(
    project_id=config.project_id,
    column=column,
    corpus_size=len(documents),
    topic_count=interpreter.topic_count,
  )
  cached_visualization_embeddings = vis_umap_model.load_cached_embeddings()
  if cached_visualization_embeddings is not None:
    return vis_umap_model.separate_embeddings(cached_visualization_embeddings)

  topic_embeddings = interpreter.topic_embeddings
  high_dimensional_embeddings = np.vstack([embeddings, topic_embeddings])
  visualization_embeddings = vis_umap_model.fit_transform(high_dimensional_embeddings)
  task.log_success(f"Finished mapping the document and topic vectors to 2D. The embeddings have been stored in {vis_umap_model.embedding_path}.")
  return vis_umap_model.separate_embeddings(visualization_embeddings)

def bertopic_post_processing(df: pd.DataFrame, intermediate: _BERTopicColumnIntermediateResult)->TopicModelingResult:
  column = intermediate.column
  model = intermediate.model
  task = intermediate.task
  interpreter = BERTopicInterpreter(intermediate.model)

  task.log_pending(f"Applying post-processing on the topics of \"{intermediate.column.name}\"...")

  # Set topic assignments
  document_topic_mapping_column = pd.Series(np.full(len(df), -1), dtype=np.int32)
  document_topic_mapping_column[intermediate.mask] = model.topics_
  document_topic_mapping_column[~intermediate.mask] = pd.NA
  df[column.topic_column.name] = document_topic_mapping_column

  # Perform hierarchical clustering
  task.log_pending(f"Calculating the topic hierarchy of \"{intermediate.column.name}\"...")
  topics = bertopic_topic_hierarchy(intermediate)
  task.log_pending(f"Calculating the topic hierarchy of \"{intermediate.column.name}\"...")

  # Embed document/topics
  bertopic_visualization_embeddings(intermediate).topic_embeddings

  # Create topic result
  topic_modeling_result = TopicModelingResult(
    project_id=intermediate.config.project_id,
    topics=topics,
    valid_count=len(intermediate.documents),
    total_count=len(intermediate.mask),
    invalid_count=len(intermediate.mask) - intermediate.mask.sum(),
    outlier_count=(document_topic_mapping_column == -1).sum()
  )

  task.log_success(f"Finished appling post-processing on the topics of \"{intermediate.column.name}\".")
  return topic_modeling_result

  