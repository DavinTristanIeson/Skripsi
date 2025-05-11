import http

import numpy as np
import pandas as pd
from routes.topic.model import DocumentTopicsVisualizationResource, DocumentVisualizationResource, TopicVisualizationResource
from modules.api.wrapper import ApiError, ApiResult
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache
from modules.topic.model import TopicModelingResult

logger = ProvisionedLogger().provision("Topic Controller")

def _assert_visualization_vectors_synced_with_workspace_df(cache: ProjectCache, column: TextualSchemaColumn):
  visualization_vectors = cache.visualization_vectors.load(column.name)
  df = cache.workspaces.load()

  if column.preprocess_column.name not in df.columns:
    raise ApiError(f"It seems that the preprocessed documents no longer exists even though you should have ran the topic modeling algorithm before. There may have been a data corruption, please run the topic modeling algorithm again or reload the whole dataset if the problem persists.", http.HTTPStatus.BAD_REQUEST)

  mask = df[column.preprocess_column.name].notna()
  document_topic_assignments = df.loc[mask, column.topic_column.name]
  documents = df.loc[mask, column.name]

  if mask.sum() != visualization_vectors.shape[0]:
    raise ApiError("The topic modeling results are not in sync with the document visualization results. The file may be corrupted. Try running the topic modeling procedure again.", http.HTTPStatus.BAD_REQUEST)
  
  return visualization_vectors, document_topic_assignments, documents

def _calculate_topic_visualization_vectors(visualization_vectors: np.ndarray, document_topic_assignments: pd.Series, tm_result: TopicModelingResult):
  logger.debug(f"Calculating topic vectors from document vectors.")
  raw_topic_vectors: list[np.ndarray] = []

  for topic in tm_result.topics:
    mask = document_topic_assignments == topic.id
    # This shouldn't happen, but just in case.
    if mask.sum() == 0:
      raw_topic_vectors.append(np.array([0, 0]))
      continue
    mean_document_vector = visualization_vectors[document_topic_assignments == topic.id].mean(axis=0)
    raw_topic_vectors.append(mean_document_vector)
  topic_vectors = np.array(raw_topic_vectors)
  return topic_vectors


def get_topic_visualization_results(cache: ProjectCache, column: TextualSchemaColumn, topic_modeling_result: TopicModelingResult):
  visualization_vectors, document_topic_assignments, documents = _assert_visualization_vectors_synced_with_workspace_df(cache, column)

  topic_vectors = _calculate_topic_visualization_vectors(visualization_vectors, document_topic_assignments, topic_modeling_result)

  if topic_vectors.shape[0] == 0:
    return ApiResult(data=[], message=None)
  
  topic_visualizations: list[TopicVisualizationResource] = []
  for i in range(topic_vectors.shape[0]):
    topic = topic_modeling_result.topics[i]
    topic_vector = topic_vectors[i] 
    topic_visualization = TopicVisualizationResource(
      frequency=topic.frequency,
      topic=topic,
      x=topic_vector[0],
      y=topic_vector[1],
    )
    topic_visualizations.append(topic_visualization)

  return ApiResult(
    data=topic_visualizations,
    message=None
  )


MAX_DOCUMENTS = 5000
def get_document_visualization_results(cache: ProjectCache, column: TextualSchemaColumn, topic_modeling_result: TopicModelingResult):
  visualization_vectors, document_topic_assignments, documents = _assert_visualization_vectors_synced_with_workspace_df(cache, column)
  topic_visualization = get_topic_visualization_results(cache, column, topic_modeling_result).data
  
  document_visualizations: list[DocumentVisualizationResource] = []

  sampled_count = min(MAX_DOCUMENTS, len(document_topic_assignments))
  unsampled_count = len(document_topic_assignments) - sampled_count
  if unsampled_count > 0:
    document_mask = np.hstack([
      np.full(sampled_count, True),
      np.full(unsampled_count, False)
    ])
    # https://stackoverflow.com/questions/19597473/binary-random-array-with-a-specific-proportion-of-ones
    np.random.seed(MAX_DOCUMENTS)
    np.random.shuffle(document_mask)
    document_topic_assignments = document_topic_assignments[document_mask]
    documents = documents[document_mask]
    visualization_vectors = visualization_vectors[document_mask, :]
    
  for topic_id, document, document_vector in zip(document_topic_assignments, documents, visualization_vectors):
    document_visualization = DocumentVisualizationResource(
      topic=topic_id,
      document=document,
      x=document_vector[0],
      y=document_vector[1],
    )
    document_visualizations.append(document_visualization)

  return ApiResult(
    data=DocumentTopicsVisualizationResource(
      topics=topic_visualization,
      documents=document_visualizations,
    ),
    message=None
  )

__all__ = [
  "get_topic_visualization_results",
  "get_document_visualization_results",
]