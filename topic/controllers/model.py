import os
from typing import cast
import bertopic
import bertopic.dimensionality
import bertopic.representation
import bertopic.vectorizers
import hdbscan
import numpy as np
import pandas as pd
import sklearn.feature_extraction

from common.ipc.requests import IPCRequestData
from common.ipc.responses import IPCResponseData
from common.ipc.task import IPCTask, TaskStepTracker
from common.logger import RegisteredLogger, TimeLogger
from wordsmith.data.config import Config
from wordsmith.data.paths import ProjectPaths
from wordsmith.data.schema import TextualSchemaColumn
from wordsmith.topic.doc2vec import Doc2VecTransformer

logger = RegisteredLogger().provision("Topic Modeling Service")

def topic_modeling(task: IPCTask):
  message = cast(IPCRequestData.TopicModeling, task.request)
  config = Config.from_project(message.project_id)

  task.progress(0, f"Loading dataset from {config.source.path}")
  df = config.source.load()
  steps = TaskStepTracker(
    max_steps=1 + len(config.data_schema.columns) + (len(config.data_schema.columns) * 4)
  )
  for colidx, (df, column) in enumerate(config.data_schema.preprocess(df)):
    df = df
    task.progress(steps.advance(), f"Preprocessing column: {column.name} with type \"{column.type}\"." +
      " Preprocessing text may take some time..." +
      f" ({colidx + 1} / {len(config.data_schema.columns)})"
    )

  result_path = config.paths.full_path(ProjectPaths.Workspace)
  df.to_parquet(result_path)
  task.progress(steps.advance(), f"Saved workspace table to {config.source.path}. You should be able to access the Table page to explore your dataset, but the topics have not been processed yet.")
  logger.info(f"Saved intermediate results to {result_path}")

  textcolumns = config.data_schema.textual()
  for colidx, column in enumerate(textcolumns):
    column_progress = f"({colidx + 1} / {len(textcolumns)})"
    column_data = df[column.preprocess_column]
    mask = column_data.str.len() != 0
    documents = cast(list[str], column_data[mask])

    task.check_stop()
    task.progress(
      progress=steps.advance(),
      message=f"Transforming documents of {column.name} into document embeddings {column_progress}"
    )

    with TimeLogger(logger, "Fitting doc2vec", report_start=True):
      doc2vec = Doc2VecTransformer()
      doc2vec.fit(documents)
      embeddings = doc2vec.transform(documents)

    kwargs = dict()
    if column.topic_modeling.max_topics is not None:
      kwargs["nr_topics"] = column.topic_modeling.max_topics
    if column.topic_modeling.seed_topics is not None:
      kwargs["seed_topic_list"] = column.topic_modeling.seed_topics


    hdbscan_kwargs = dict()
    if column.topic_modeling.max_topic_size is not None:
      hdbscan_kwargs["max_cluster_size"] = int(column.topic_modeling.max_topic_size * len(documents))
    
    hdbscan_model = hdbscan.HDBSCAN(
      min_cluster_size=column.topic_modeling.min_topic_size,
      metric="euclidean",
      cluster_selection_method="eom",
      prediction_data=True,
      **hdbscan_kwargs
    )

    ctfidf_model = bertopic.vectorizers.ClassTfidfTransformer(
      bm25_weighting=True,
      reduce_frequent_words=True,
    )

    vectorizer_model = sklearn.feature_extraction.text.CountVectorizer(min_df=5, max_df=0.5, ngram_range=column.topic_modeling.n_gram_range)

    model = bertopic.BERTopic(
      embedding_model=doc2vec,
      hdbscan_model=hdbscan_model,
      ctfidf_model=ctfidf_model,
      vectorizer_model=vectorizer_model,
      representation_model=bertopic.representation.MaximalMarginalRelevance(),
      low_memory=column.topic_modeling.low_memory,
      calculate_probabilities=False,
      verbose=True,
      **kwargs,
    )

    task.check_stop()
    task.progress(
      progress=steps.advance(),
      message=f"Starting the topic modeling process for {column.name} {column_progress}"
    )

    with TimeLogger(logger, "Performing Topic Modeling", report_start=True):
      topics, probs = model.fit_transform(documents, embeddings)

    task.check_stop()
    task.progress(
      progress=steps.advance(),
      message=f"Finished the topic modeling process for {column.name}. Performing additional post-processing for the discovered topics. {column_progress}"
    )

    if column.topic_modeling.no_outliers:
      topics = model.reduce_outliers(documents, topics, strategy="embeddings", embeddings=embeddings)
      if column.topic_modeling.represent_outliers:
        model.update_topics(documents, topics=topics)

    topic_number_column = pd.Series(np.full((len(df[column.name],)), -1), dtype=np.int32)
    topic_number_column[mask] = topics

    topic_column = pd.Categorical(topic_number_column)
    topic_column = topic_column.rename_categories({**model.topic_labels_, -1: pd.NA})
    df.loc[:, column.topic_column] = topic_column

    task.check_stop()
    task.progress(
      progress=steps.advance(),
      message=f"Saving the topic information for {column.name} {column_progress}"
    )

    logger.info(f"Topics of {column.name}: {model.topic_labels_}. ")

    doc2vec_path = config.paths.full_path(os.path.join(ProjectPaths.Doc2Vec, column.name))
    doc2vec_root_path = config.paths.full_path(os.path.join(ProjectPaths.Doc2Vec))
    if not os.path.exists(doc2vec_root_path):
      os.makedirs(doc2vec_root_path)
      logger.info(f"Created {doc2vec_root_path} since it hasn't existed before.")
    doc2vec.model.save(doc2vec_path)
    
    bertopic_path = config.paths.full_path(os.path.join(ProjectPaths.BERTopic, column.name))
    bertopic_root_path = config.paths.full_path(os.path.join(ProjectPaths.BERTopic))
    if not os.path.exists(bertopic_root_path):
      os.makedirs(bertopic_root_path)
      logger.info(f"Created {bertopic_root_path} since it hasn't existed before.")
    model.save(bertopic_path, "safetensors", save_ctfidf=True)
      
  task.check_stop()
  df.to_parquet(result_path)
  task.success(IPCResponseData.Empty(), message=f"Finished discovering topics in Project \"{task.request.project_id}\" (data sourced from {config.source.path})")

__all__ = [
  "topic_modeling"
]