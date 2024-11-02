from typing import Sequence, cast

import sklearn.metrics
import pydantic

from common.ipc.requests import IPCRequestData
from common.ipc.responses import IPCResponseData
from common.ipc.task import IPCTask, TaskStepTracker
from common.models.api import ApiError
from topic.controllers.utils import assert_column_exists
from wordsmith.data.config import Config
from wordsmith.data.schema import SchemaColumnTypeEnum
from wordsmith.topic.interpret import bertopic_topic_labels
from wordsmith.visual import bertopicvis
import scipy.sparse
import numpy as np

# Metric used by BERTopic
from sklearn.metrics.pairwise import cosine_similarity

class TemporaryTopicAssignment(pydantic.BaseModel):
  topic: int
  parent_id: int
  name: str
  amidst: int

def hierarchical_topics_distance_matrix(x: scipy.sparse.csr_matrix)->scipy.sparse.csr_matrix:
  # The values end up as negative for some reason, so this fix is necessary
  # https://github.com/MaartenGr/BERTopic/issues/1137
  # Clamp all values to 0.
  return np.max(0, 1 - cosine_similarity(x))

def hierarchical_topic_plot(task: IPCTask):
  steps = TaskStepTracker(
    max_steps = 4,
  )
  message = cast(IPCRequestData.Topics, task.request)

  config = Config.from_project(message.project_id)
  column = config.data_schema.assert_exists(message.column)
  if column.type != SchemaColumnTypeEnum.Textual:
    raise ApiError("We can only extract topics from textual columns.", 400)

  task.progress(0, f"Loading topic information for {message.column}.")
  model = config.paths.load_bertopic(message.column)
  
  task.progress(steps.advance(), f"Loading workspace table.")
  df = config.paths.load_workspace()

  column_data = assert_column_exists(df, column.preprocess_column)
  mask = column_data.str.len() != 0
  documents = cast(list[str], column_data[mask])

  task.progress(steps.advance(), f"Calculating topic hierarchy for {message.column}.")
  if model.c_tf_idf_.shape[0] <= 1: #type: ignore
    task.error(ValueError(f"It looks like the topic modeling procedure failed to find any topics for {message.column}. It might be because there's too few documents to train the model or an imprroper configuration."))
  hierarchical_topics = model.hierarchical_topics(documents, distance_function=hierarchical_topics_distance_matrix)

  task.progress(steps.advance(), f"Visualizing topic hierarchy for {message.column}.")
  # print(hierarchical_topics, model.topic_labels_.keys())
  fig = model.visualize_hierarchy(hierarchical_topics=hierarchical_topics, title=f"Topics of {message.column}")

  # fig = bertopicvis.hierarchical_topics_sunburst(
  #   hierarchical_topics,
  #   model.topic_labels_, # type: ignore
  #   model.topic_sizes_ # type: ignore
  # ) 

  topic_words_dict = cast(dict[str, Sequence[tuple[str, float]]], model.get_topics())
  if -1 in topic_words_dict:
    # Remove outliers
    topic_words_dict.pop(-1) # type: ignore
  topic_words = list(topic_words_dict.values())
  
  frequencies: list[int] = []
  topics = bertopic_topic_labels(model, outliers=False)
  for id in range(len(topics)):
    frequencies.append(cast(int, model.get_topic_freq(id)))

  outliers = cast(int, model.get_topic_freq(-1))
  total = outliers + sum(frequencies)
  
  task.success(IPCResponseData.Topics(
    plot=str(fig.to_json()),
    topics=topics,
    topic_words=topic_words,
    frequencies=frequencies,
    outliers=outliers,
    total=total,
    column=message.column
  ), None)

def topic_similarity_plot(task: IPCTask):
  steps = TaskStepTracker(
    max_steps = 2,
  )
  message = cast(IPCRequestData.TopicSimilarityPlot, task.request)

  config = Config.from_project(message.project_id)

  task.progress(0, f"Loading topic information for {message.column}.")
  model = config.paths.load_bertopic(message.column)

  task.progress(steps.advance(), f"Calculating topic correlation for {message.column}.")

  # Heatmap
  heatmap = model.visualize_heatmap(title=f"{message.column.capitalize()} Topic Similarity Matrix")
  # Topic Distribution
  ldavis = model.visualize_topics(title=f"{message.column.capitalize()} Topics Distribution")


  topics = bertopic_topic_labels(model, outliers=False)
  similarity_matrix_raw = sklearn.metrics.pairwise.cosine_similarity(model.topic_embeddings_) # type: ignore
  
  similarity_matrix: list[list[float]] = []
  for row in similarity_matrix_raw:
    similarity_matrix.append(list(row))
  
  task.success(IPCResponseData.TopicSimilarity(
    topics=topics,
    ldavis=cast(str, ldavis.to_json()),
    heatmap=cast(str, heatmap.to_json()),
    similarity_matrix=similarity_matrix,
    column=message.column
  ), None)
  
__all__ = [
  "topic_similarity_plot",
  "hierarchical_topic_plot",
]