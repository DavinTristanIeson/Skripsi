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
from common.logger import TimeLogger
from wordsmith.data.config import Config
from wordsmith.data.paths import ProjectPaths
from wordsmith.topic.doc2vec import Doc2VecTransformer

logger = logging.getLogger("Topic Modeling Service")
def topic_modeling(message: IPCRequestData.TopicModeling):
  config = Config.from_project(message.project_id)
  df = config.preprocess()
  textcolumns = config.dfschema.textual()
  for column in textcolumns:
    column_data = df[column.preprocess_column]
    mask = column_data.str.len() != 0
    raw_documents = cast(Sequence[str], column_data[mask])

    with TimeLogger(logger, "Preprocessing Documents", report_start=True):
      tokens: Sequence[Sequence[str]]
      tokens = tuple(column.preprocessing.preprocess(
        cast(Sequence[str], raw_documents)
      ))
      documents = list(common.utils.loader.concatenate_generator(tokens))

    doc2vec = Doc2VecTransformer()

    doc2vec.fit(documents)
    embeddings = doc2vec.transform(documents)

    kwargs = dict()
    if column.topic.max_topics is not None:
      kwargs["nr_topics"] = column.topic.max_topics
    if column.topic.seed_topics is not None:
      kwargs["seed_topic_list"] = column.topic.seed_topics

    max_topic_size = int(column.topic.max_topic_size * len(documents)) \
      if isinstance(column.topic.max_topic_size, float) \
      else column.topic.max_topic_size
    
    hdbscan_model = hdbscan.HDBSCAN(
      min_cluster_size=column.topic.min_topic_size,
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
      low_memory=column.topic.low_memory,
      min_topic_size=column.topic.min_topic_size,
      n_gram_range=column.topic.n_gram_range,
      calculate_probabilities=True,
      **kwargs,
    )

    with TimeLogger(logger, "Performing Topic Modeling", report_start=True):
      topics, probs = model.fit_transform(documents, embeddings)

    if column.topic.no_outliers:
      topics = model.reduce_outliers(documents, topics, strategy="embeddings", embeddings=embeddings)
      if column.topic.represent_outliers:
        model.update_topics(documents, topics=topics)

    topic_number_column = pd.Series(np.full((len(raw_documents,)), -1), dtype=np.int32)
    topic_number_column[mask] = topics

    topic_column = pd.Categorical(topic_number_column)
    topic_column.rename_categories({**model.topic_labels_, -1: -1})
    df[column.topic_column] = topic_column

    doc2vec_path = config.paths.full_path(os.path.join(ProjectPaths.Doc2Vec, column.name))
    doc2vec.model.save(doc2vec_path)

    bertopic_path = config.paths.full_path(os.path.join(ProjectPaths.BERTopic, column.name))
    model.save(bertopic_path, "safetensors")

  df.to_parquet()