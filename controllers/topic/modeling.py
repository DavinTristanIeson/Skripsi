import numpy as np
import pandas as pd

from common.logger import TimeLogger
from common.task.executor import TaskPayload
from controllers.topic.interpret import bertopic_topic_labels
from controllers.topic.utils import BERTopicColumnIntermediateResult
from models.config.config import Config
from models.config.paths import ProjectPaths
from models.config.schema import TextualSchemaColumn

def topic_modeling(
  task: TaskPayload,
  config: Config,
  intermediate: BERTopicColumnIntermediateResult,
  index: int,
  total: int
):
  import bertopic
  import bertopic.vectorizers
  import bertopic.representation
  import hdbscan
  import sklearn.feature_extraction
  from umap import UMAP

  column = intermediate.column
  documents = intermediate.documents
  mask = intermediate.mask
  embeddings = intermediate.embeddings

  kwargs = dict()
  if column.topic_modeling.max_topics is not None:
    kwargs["nr_topics"] = column.topic_modeling.max_topics

  max_cluster_size = int(column.topic_modeling.max_topic_size * len(documents))

  if column.topic_modeling.min_topic_size >= max_cluster_size:
    raise ValueError("Min. topic size should not be greater than max. topic size. Please set a higher max topic size. Note: This can also happen if you have too few valid documents to analyze.")
  
  umap_model = UMAP(
    n_neighbors=column.topic_modeling.globality_consideration or column.topic_modeling.min_topic_size,
    min_dist=0.1,
    # BERTopic uses 5 dimensions
    n_components=5,
    metric="euclidean",
    low_memory=column.topic_modeling.low_memory
  )

  hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=column.topic_modeling.min_topic_size,
    max_cluster_size=max_cluster_size,
    min_samples=max(2, int(column.topic_modeling.clustering_conservativeness * column.topic_modeling.min_topic_size)),
    metric="euclidean",
    cluster_selection_method="eom",
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
    umap_model=umap_model,
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
  task.progress(f"Starting the topic modeling process for {column.name}")

  with TimeLogger("Topic Modeling", "Performing Topic Modeling", report_start=True):
    topics, probs = model.fit_transform(documents, embeddings)


  task.check_stop()
  task.progress(f"Finished the topic modeling process for {column.name}. Performing additional post-processing for the discovered topics.")

  umap_embeddings_path = config.paths.full_path(ProjectPaths.Embeddings)
  task.progress(f"Saving the UMAP embeddings for {column.name} to ")

  if column.topic_modeling.no_outliers:
    topics = model.reduce_outliers(documents, topics, strategy="embeddings", embeddings=embeddings)
    if column.topic_modeling.represent_outliers:
      model.update_topics(intermediate.documents, topics=topics)

  topic_labels = bertopic_topic_labels(model)
  if model._outliers:
    topic_labels.insert(0, '')
  model.set_topic_labels(topic_labels)

  topic_number_column = pd.Series(np.full(len(intermediate.documents), -1), dtype=np.int32)
  topic_number_column[intermediate.mask] = topics

  labels_mapper = {k: v for k, v in enumerate(topic_labels)}

  topic_column = topic_number_column.replace({**labels_mapper, -1: ''})
  topic_column = pd.Categorical(topic_column)