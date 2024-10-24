import logging
import os
from typing import Sequence, cast
import bertopic
import bertopic.dimensionality
import bertopic.representation
import bertopic.vectorizers
import hdbscan
import numpy as np
import pandas as pd

import common
from common.ipc.requests import IPCRequestData
from common.ipc.responses import IPCResponseData
from common.ipc.task import IPCTask, TaskStepTracker
from common.logger import TimeLogger
from wordsmith.data.config import Config
from wordsmith.data.paths import ProjectPaths
from wordsmith.topic.doc2vec import Doc2VecTransformer

logger = logging.getLogger("Topic Modeling Service")

def topic_modeling(task: IPCTask):
  message = cast(IPCRequestData.TopicModeling, task.request)
  config = Config.from_project(message.project_id)

  task.progress(0, "Preprocessing all of the available columns")
  df = config.preprocess()

  steps = TaskStepTracker(
    max_steps=1 + (len(config.dfschema.columns) * 5)
  )
  textcolumns = config.dfschema.textual()
  for colidx, column in enumerate(textcolumns):
    column_data = df[column.preprocess_column]
    mask = column_data.str.len() != 0
    raw_documents = cast(Sequence[str], column_data[mask])

    task.check_stop()
    task.progress(
      progress=steps.advance(),
      message=f"Preprocessing documents of {column.name}"
    )

    with TimeLogger(logger, "Preprocessing Documents", report_start=True):
      tokens: Sequence[Sequence[str]]
      tokens = tuple(column.preprocessing.preprocess(
        cast(Sequence[str], raw_documents)
      ))
      documents = list(common.utils.loader.concatenate_generator(tokens))

    task.check_stop()
    task.progress(
      progress=steps.advance(),
      message=f"Transforming documents of {column.name} into document embeddings"
    )

    doc2vec = Doc2VecTransformer()
    doc2vec.fit(documents)
    embeddings = doc2vec.transform(documents)

    kwargs = dict()
    if column.topic_modeling.max_topics is not None:
      kwargs["nr_topics"] = column.topic_modeling.max_topics
    if column.topic_modeling.seed_topics is not None:
      kwargs["seed_topic_list"] = column.topic_modeling.seed_topics

    max_topic_size = int(column.topic_modeling.max_topic_size * len(documents)) \
      if isinstance(column.topic_modeling.max_topic_size, float) \
      else column.topic_modeling.max_topic_size
    
    hdbscan_model = hdbscan.HDBSCAN(
      min_cluster_size=column.topic_modeling.min_topic_size,
      max_cluster_size=max_topic_size,
      metric="euclidean",
      cluster_selection_method="eom",
      prediction_data=True,
    )

    ctfidf_model = bertopic.vectorizers.ClassTfidfTransformer(
      bm25_weighting=True,
      reduce_frequent_words=True,
    )

    model = bertopic.BERTopic(
      embedding_model=doc2vec,
      hdbscan_model=hdbscan_model,
      ctfidf_model=ctfidf_model,
      representation_model=bertopic.representation.MaximalMarginalRelevance(),
      low_memory=column.topic_modeling.low_memory,
      min_topic_size=column.topic_modeling.min_topic_size,
      n_gram_range=column.topic_modeling.n_gram_range,
      calculate_probabilities=True,
      **kwargs,
    )

    task.check_stop()
    task.progress(
      progress=steps.advance(),
      message=f"Starting the topic modeling process for {column.name}"
    )

    with TimeLogger(logger, "Performing Topic Modeling", report_start=True):
      topics, probs = model.fit_transform(documents, embeddings)

    task.check_stop()
    task.progress(
      progress=steps.advance(),
      message=f"Finished the topic modeling process for {column.name}. Performing additional post-processing for the discovered topics."
    )

    if column.topic_modeling.no_outliers:
      topics = model.reduce_outliers(documents, topics, strategy="embeddings", embeddings=embeddings)
      if column.topic_modeling.represent_outliers:
        model.update_topics(documents, topics=topics)

    topic_number_column = pd.Series(np.full((len(raw_documents,)), -1), dtype=np.int32)
    topic_number_column[mask] = topics

    topic_column = pd.Categorical(topic_number_column)
    topic_column.rename_categories({**model.topic_labels_, -1: -1})
    df[column.topic_column] = topic_column

    task.check_stop()
    task.progress(
      progress=steps.advance(),
      message=f"Saving the topic information for {column.name}"
    )

    doc2vec_path = config.paths.full_path(os.path.join(ProjectPaths.Doc2Vec, column.name))
    doc2vec.model.save(doc2vec_path)

    bertopic_path = config.paths.full_path(os.path.join(ProjectPaths.BERTopic, column.name))
    model.save(bertopic_path, "safetensors")
    
  task.check_stop()
  df.to_parquet()
  task.success(IPCResponseData.Empty(), message=f"Finished discovering topics in Project \"{task.request.project_id}\" (data sourced from {config.source.path})")

__all__ = [
  "topic_modeling"
]