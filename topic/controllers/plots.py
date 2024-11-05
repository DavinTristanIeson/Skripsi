from typing import Sequence, cast

import numpy as np
import sklearn.metrics
import pydantic
import pandas as pd
import plotly.express

from common.ipc.requests import IPCRequestData
from common.ipc.responses import IPCResponse, IPCResponseData, IPCResponseStatus
from common.ipc.task import IPCTask, TaskStepTracker
from common.logger import RegisteredLogger
from common.utils.string import truncate_strings
from common.models.api import ApiError
from topic.controllers.utils import assert_column_exists
from wordsmith.data.config import Config
from wordsmith.data.schema import SchemaColumnTypeEnum
from wordsmith.topic.interpret import bertopic_topic_labels

# Metric used by BERTopic
from sklearn.metrics.pairwise import cosine_distances

class TemporaryTopicAssignment(pydantic.BaseModel):
  topic: int
  parent_id: int
  name: str
  amidst: int

logger = RegisteredLogger().provision("Topic Modeling Service")
def topic_plot(task: IPCTask):
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

  topic_words_dict = cast(dict[str, Sequence[tuple[str, float]]], model.get_topics())
  if -1 in topic_words_dict:
    # Remove outliers
    topic_words_dict.pop(-1) # type: ignore
  topic_words = list(topic_words_dict.values())
  
  topic_frequencies: list[int] = []
  topics = bertopic_topic_labels(model)
  for id in range(len(topics)):
    topic_frequencies.append(cast(int, model.get_topic_freq(id)))

  try:
    outliers = cast(int, model.get_topic_freq(-1))
  except KeyError:
    outliers = 0

  task.progress(steps.advance(), f"Visualizing all of the keywords of the topics discovered in {message.column}.")
  topics_barchart = model.visualize_barchart(top_n_topics=100000000, n_words=10, title=f"Topic Keywords of {message.column}", autoscale=True)

  task.progress(steps.advance(), f"Visualizing the topic frequencies of {message.column} as a barchart.")
  frequencies_df = pd.DataFrame({
    "Topic": tuple(truncate_strings(topics)),
    "Frequency": topic_frequencies,
  })
  normalized_frequency: pd.Series = (frequencies_df["Frequency"] / frequencies_df["Frequency"].sum()) * 100
  percentages = normalized_frequency.map(lambda x: f"{x:.2f}")
  frequency_barchart = plotly.express.bar(frequencies_df, orientation="h", x="Frequency", y="Topic")
  frequency_barchart.update_traces(
    customdata=tuple(zip(topics, percentages)),
    hovertemplate="<br>".join([
      "Topic: %{customdata[0]}",
      "Frequency: %{x} (%{customdata[1]}%)",
    ])
  )
  
  task.success(IPCResponseData.Topics(
    topics=topics,
    topic_words=topic_words,
    frequencies=topic_frequencies,
    outliers=outliers,
    total=len(column_data),
    valid=sum(topic_frequencies),
    invalid=int(np.size(mask) - np.count_nonzero(mask)),
    column=message.column,
    frequency_barchart=cast(str, frequency_barchart.to_json()),
    topics_barchart=cast(str, topics_barchart.to_json())
  ), None)

def topic_similarity_plot(task: IPCTask):
  steps = TaskStepTracker(
    max_steps = 9,
  )
  message = cast(IPCRequestData.TopicSimilarityPlot, task.request)

  config = Config.from_project(message.project_id)
  column = config.data_schema.assert_exists(message.column)
  if column.type != SchemaColumnTypeEnum.Textual:
    raise ApiError("We can only extract topics from textual columns.", 400)

  task.progress(0, f"Loading topic information for {message.column}.")
  model = config.paths.load_bertopic(message.column)

  task.progress(steps.advance(), f"Loading workspace table.")
  df = config.paths.load_workspace()
  column_data = assert_column_exists(df, column.preprocess_column)
  mask = column_data != ''
  documents = cast(list[str], column_data[mask])

  task.progress(steps.advance(), f"Loading document embeddings.")
  embeddings = config.paths.load_embeddings(message.column)

  topic_labels = list(truncate_strings(bertopic_topic_labels(model)))
  if model._outliers:
    topic_labels.insert(0, "Outlier")
  model.set_topic_labels(topic_labels)

  if model.c_tf_idf_ is not None and model.c_tf_idf_.shape[0] <= 1: #type: ignore
    task.error(ValueError(f"It looks like the topic modeling procedure failed to find any topics for {message.column}. It might be because there's too few documents to train the model or an improper preprocessing configuration."))
    return

  task.progress(steps.advance(), f"Calculating topic similarity for {message.column}.")
  similarity_matrix_raw = sklearn.metrics.pairwise.cosine_similarity(model.c_tf_idf_) # type: ignore
  # Convert numpy array to python dtype
  similarity_matrix: list[list[float]] = []
  for row in similarity_matrix_raw:
    similarity_matrix.append(list(row))

  task.progress(steps.advance(), f"Creating topic similarity heatmap for {message.column}.")
  # Heatmap
  heatmap = model.visualize_heatmap(title=f"{message.column} Topic Similarity Matrix", custom_labels=True)

  task.progress(steps.advance(), f"Creating LDAvis-style visualization for {message.column}.")
  try:
    # LDAvis
    ldavis = model.visualize_topics(title=f"{message.column} Topics Distribution", custom_labels=True)
  except Exception as e:
    # This may happen if there's too few documents. https://github.com/MaartenGr/BERTopic/issues/97
    logger.error(f"An error has occurred while creating the LDAvis visualization of {message.column}. Error => {e}")
    ldavis = None

  # Dendrogram
  task.progress(steps.advance(), f"Calculating topic hierarchy for {message.column}.")
  hierarchical_topics = model.hierarchical_topics(
    documents,
    distance_function=cosine_distances # type: ignore
  )

  task.progress(steps.advance(), f"Visualizing topic hierarchy for {message.column}.")
  dendrogram = model.visualize_hierarchy(hierarchical_topics=hierarchical_topics, title=f"Topics of {message.column}", custom_labels=True)

  topics = bertopic_topic_labels(model)

  task.progress(steps.advance(), f"Visualizing the documents of {message.column} and their assigned topics as a scatterplot.")
  original_documents: list[str] = list(df.loc[mask, message.column])
  topic_indices: list[int] = list(df.loc[mask, column.topic_index_column])
  scatterplot = model.visualize_documents(original_documents, topic_indices, embeddings=embeddings, title=f"Documents of {message.column}", custom_labels=True)

  with task.lock:
    task.results[task.id] = IPCResponse(
      id=task.id,
      data=IPCResponseData.TopicSimilarity(
        topics=topics,
        ldavis=cast(str, ldavis.to_json()) if ldavis is not None else '',
        heatmap=cast(str, heatmap.to_json()),
        similarity_matrix=similarity_matrix,
        dendrogram=cast(str, dendrogram.to_json()),
        scatterplot=cast(str, scatterplot.to_json()),
        column=message.column,
      ),
      message="We are unable to create the LDAvis-style visualization because of there are too few documents/topics."
        if ldavis is None
        else "The relationship between topics as well as the key terms that make up each topic has been successfully visualized.",
      progress=1,
      status=IPCResponseStatus.Success,
    )
  
__all__ = [
  "topic_similarity_plot",
  "topic_plot",
]