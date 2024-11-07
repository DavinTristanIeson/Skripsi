import itertools
import os
from typing import Optional, cast
import bertopic
import bertopic.dimensionality
import bertopic.representation
import bertopic.vectorizers
import hdbscan
import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.decomposition
import sklearn.feature_extraction
import sklearn.feature_extraction.text
from sklearn.pipeline import make_pipeline

from common.ipc.requests import IPCRequestData
from common.ipc.responses import IPCResponseData
from common.ipc.task import IPCTask, TaskStepTracker
from common.logger import RegisteredLogger, TimeLogger
from common.models.api import ApiError
from wordsmith.data.config import Config
from wordsmith.data.paths import ProjectPaths
from wordsmith.data.textual import DocumentEmbeddingMethodEnum
from wordsmith.topic.doc2vec import Doc2VecTransformer
from wordsmith.topic.interpret import bertopic_topic_labels

logger = RegisteredLogger().provision("Topic Modeling Service")

def topic_modeling(task: IPCTask):
  message = cast(IPCRequestData.TopicModeling, task.request)
  config = Config.from_project(message.project_id)

  TOPIC_MODELING_STEPS = 4
  result_path = config.paths.full_path(ProjectPaths.Workspace)
  if os.path.exists(result_path):
    task.progress(0, f"Loading cached dataset from {result_path}")
    steps = TaskStepTracker(
      max_steps=1 + (len(config.data_schema.columns) * TOPIC_MODELING_STEPS)
    )
    df = config.paths.load_workspace()
  else:
    task.progress(0, f"Loading dataset from {config.source.path}")
    df = config.source.load()
    steps = TaskStepTracker(
      max_steps=1 + len(config.data_schema.columns) + (len(config.data_schema.columns) * TOPIC_MODELING_STEPS)
    )
    for colidx, (df, column) in enumerate(config.data_schema.preprocess(df)):
      task.check_stop()
      df = df
      task.progress(steps.advance(), f"Preprocessing column: {column.name} with type \"{column.type}\"." +
        " Preprocessing text may take some time..." +
        f" ({colidx + 1} / {len(config.data_schema.columns)})"
      )
    df.to_parquet(result_path)

  task.progress(steps.advance(), f"Saved workspace table to {config.source.path}. You should be able to access the Table page to explore your dataset, but the topics have not been processed yet.")
  logger.info(f"Saved intermediate results to {result_path}")

  textcolumns = config.data_schema.textual()
  for colidx, column in enumerate(textcolumns):
    column_progress = f"({colidx + 1} / {len(textcolumns)})"
    column_data = df[column.preprocess_column]
    mask = column_data.str.len() != 0
    documents: list[str] = list(column_data[mask])

    if len(documents) == 0:
      raise ValueError(f"{column.name} does not contain any valid documents after the preprocessing step. Either change the preprocessing configuration of {column.name} to be more lax (e.g: lower the min word frequency, min document length), or set the type of this column to Unique.")

    task.check_stop()
    task.progress(
      progress=steps.advance(),
      message=f"Transforming documents of {column.name} into document embeddings {column_progress}"
    )

    if column.topic_modeling.embedding_method == DocumentEmbeddingMethodEnum.Doc2Vec:
      doc2vec: Optional[Doc2VecTransformer]
      with TimeLogger(logger, "Fitting doc2vec", report_start=True):
        doc2vec = Doc2VecTransformer()
        doc2vec.fit(documents)
        embeddings = doc2vec.transform(documents)
      embedding_model = doc2vec
    elif column.topic_modeling.embedding_method == DocumentEmbeddingMethodEnum.SBERT:
      with TimeLogger(logger, "Creating embeddings from SBERT", report_start=True):
        try:
          from sentence_transformers import SentenceTransformer
        except ImportError:
          raise ApiError("The sentence_transformers library must be installed before SBERT document embedding can be performed.", 400)
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        original_documents: list[str] = list(df.loc[mask, column.name])
        # Use the original documents since SBERT performs better with full data
        embeddings = sbert.encode(original_documents, show_progress_bar=True)
      embedding_model = sbert
    elif column.topic_modeling.embedding_method == DocumentEmbeddingMethodEnum.TFIDF:
      with TimeLogger(logger, "Fitting TF-IDF Vectorizer", report_start=True):
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
          min_df=1,
          max_df=1,
          ngram_range=(1,1),
          stop_words=None,
        )
        pipeline: list[sklearn.base.BaseEstimator] = [vectorizer]
        sparse_embeddings = vectorizer.fit_transform(documents)
        if sparse_embeddings.shape[1] > 100:
          svd = sklearn.decomposition.TruncatedSVD(100)
          embeddings = svd.fit_transform(sparse_embeddings)
          pipeline.append(svd)
        else:
          embeddings: npt.NDArray = sparse_embeddings.todense() # type: ignore
      embedding_model = make_pipeline(*pipeline)
    else:
      raise ApiError(f"Invalid document embedding method: {column.topic_modeling.embedding_method}", 422)

    kwargs = dict()
    if column.topic_modeling.max_topics is not None:
      kwargs["nr_topics"] = column.topic_modeling.max_topics
    if column.topic_modeling.seed_topics is not None:
      kwargs["seed_topic_list"] = column.topic_modeling.seed_topics

    max_cluster_size = int(column.topic_modeling.max_topic_size * len(documents))

    if column.topic_modeling.min_topic_size >= max_cluster_size:
      raise ValueError("Min. topic size should not be greater than max. topic size. Please set a higher max topic size. Note: This can also happen if you have too few valid documents to analyze.")
    
    hdbscan_model = hdbscan.HDBSCAN(
      min_cluster_size=column.topic_modeling.min_topic_size,
      max_cluster_size=max_cluster_size,
      metric="euclidean",
      cluster_selection_method="eom",
      prediction_data=True,
    )

    ctfidf_model = bertopic.vectorizers.ClassTfidfTransformer(
      bm25_weighting=True,
      reduce_frequent_words=True,
    )

    vectorizer_model = sklearn.feature_extraction.text.CountVectorizer(
      min_df=column.preprocessing.min_df,
      max_df=column.preprocessing.max_df,
      stop_words=None,
      ngram_range=column.topic_modeling.n_gram_range
    )

    model = bertopic.BERTopic(
      embedding_model=embedding_model,
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

    topic_labels = bertopic_topic_labels(model)
    if model._outliers:
      topic_labels.insert(0, '')
    model.set_topic_labels(topic_labels)

    topic_number_column = pd.Series(np.full((len(df[column.name],)), -1), dtype=np.int32)
    topic_number_column[mask] = topics

    labels_mapper = {k: v for k, v in enumerate(topic_labels)}

    topic_column = topic_number_column.replace({**labels_mapper, -1: ''})
    topic_column = pd.Categorical(topic_column)

    df[column.topic_column] = topic_column
    df[column.topic_index_column] = topic_number_column

    task.check_stop()
    task.progress(
      progress=steps.advance(),
      message=f"Saving the topic information for {column.name} {column_progress}"
    )

    logger.info(f"Topics of {column.name}: {model.topic_labels_}. ")

    embeddings_path = config.paths.full_path(os.path.join(ProjectPaths.Embeddings, f"{column.name}.npy"))
    embeddings_root_path = config.paths.full_path(os.path.join(ProjectPaths.Embeddings))
    if not os.path.exists(embeddings_root_path):
      os.makedirs(embeddings_root_path)
      logger.info(f"Created {embeddings_root_path} since it hasn't existed before.")
    np.save(embeddings_path, embeddings)
    
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