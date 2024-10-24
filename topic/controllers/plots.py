from typing import Sequence, cast

import sklearn.metrics
from common.ipc.requests import IPCRequestData
import plotly.express

from common.ipc.responses import IPCResponseData
from common.ipc.task import IPCTask, TaskStepTracker
from topic.controllers.utils import assert_column_exists
from wordsmith.data.config import Config

def hierarchical_topic_plot(task: IPCTask):
  steps = TaskStepTracker(
    max_steps = 4,
  )
  message = cast(IPCRequestData.Topics, task.request)

  config = Config.from_project(message.project_id)

  task.progress(0, f"Loading topic information for {message.column}.")
  model = config.paths.load_bertopic(message.column)
  
  task.progress(steps.advance(), f"Loading workspace table.")
  df = config.paths.load_workspace()

  documents = cast(list[str], assert_column_exists(df, message.column))

  task.progress(steps.advance(), f"Calculating topic hierarchy for {message.column}.")
  hierarchical_topics = model.hierarchical_topics(documents)

  task.progress(steps.advance(), f"Visualizing topic hierarchy for {message.column}.")
  sunburst = plotly.express.sunburst(
    hierarchical_topics,
    names="Topics",
    parents="Parent ID",
    values="Distance"
  )

  topic_words_dict = cast(dict[str, Sequence[tuple[str, float]]], model.get_topics())
  if -1 in topic_words_dict:
    # Remove outliers
    topic_words_dict.pop(-1) # type: ignore
  topic_words = list(topic_words_dict.values())
  
  topics: list[str] = []
  frequencies: list[int] = []
  for id, topic in enumerate(topic_words):
    topics.append(' | '.join(map(lambda x: x[0], topic[:3])))
    frequencies.append(cast(int, model.get_topic_freq(id)))

  outliers = cast(int, model.get_topic_freq(-1))
  total = outliers + sum(frequencies)
  
  task.success(IPCResponseData.Topics(
    plot=cast(str, sunburst.to_json()),
    topics=topics,
    topic_words=topic_words,
    frequencies=frequencies,
    outliers=outliers,
    total=total
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


  topic_words_dict = cast(dict[str, Sequence[tuple[str, float]]], model.get_topics())
  if -1 in topic_words_dict:
    # Remove outliers
    topic_words_dict.pop(-1) # type: ignore
  topic_words = list(topic_words_dict.values())
  
  topics: list[str] = []
  for id, topic in enumerate(topic_words):
    topics.append(' | '.join(map(lambda x: x[0], topic[:3])))

  similarity_matrix_raw = sklearn.metrics.pairwise.cosine_similarity(model.topic_embeddings_) # type: ignore
  
  similarity_matrix: list[list[float]] = []
  for row in similarity_matrix_raw:
    similarity_matrix.append(list(row))
  
  task.success(IPCResponseData.TopicSimilarity(
    topics=topics,
    ldavis=cast(str, ldavis.to_json()),
    heatmap=cast(str, heatmap.to_json()),
    similarity_matrix=similarity_matrix
  ), None)
  
__all__ = [
  "topic_similarity_plot",
  "hierarchical_topic_plot",
]