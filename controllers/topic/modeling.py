import os
from common.logger import TimeLogger
from controllers.topic.dimensionality_reduction import BERTopicCachedUMAP
from controllers.topic.utils import BERTopicColumnIntermediateResult
from models.config.paths import ProjectPaths

def bertopic_topic_modeling(
  intermediate: BERTopicColumnIntermediateResult,
):
  import bertopic
  import bertopic.vectorizers
  import bertopic.representation
  import hdbscan
  import sklearn.feature_extraction

  column = intermediate.column
  config = intermediate.config
  documents = intermediate.embedding_documents
  embeddings = intermediate.embeddings
  task = intermediate.task

  kwargs = dict()
  if column.topic_modeling.max_topics is not None:
    kwargs["nr_topics"] = column.topic_modeling.max_topics

  max_cluster_size = int(column.topic_modeling.max_topic_size * len(documents))

  if column.topic_modeling.min_topic_size >= max_cluster_size:
    raise ValueError("Min. topic size should not be greater than max. topic size. Please set a higher max topic size. Note: This can also happen if you have too few valid documents to analyze.")
  
  umap_model = BERTopicCachedUMAP(
    paths=config.paths,
    column=column,
  )

  hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=max(2, column.topic_modeling.min_topic_size),
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
    umap_model=umap_model, # type: ignore
    hdbscan_model=hdbscan_model,
    ctfidf_model=ctfidf_model,
    vectorizer_model=vectorizer_model,
    top_n_words=50,
    representation_model=bertopic.representation.MaximalMarginalRelevance(top_n_words=50),
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

  if column.topic_modeling.no_outliers:
    topics = model.reduce_outliers(documents, topics, strategy="embeddings", embeddings=embeddings)
    if column.topic_modeling.represent_outliers:
      model.update_topics(intermediate.embedding_documents, topics=topics)

  intermediate.model = model
  intermediate.topics = topics

  bertopic_path = config.paths.full_path(os.path.join(ProjectPaths.BERTopic(column.name)))
  model.save(bertopic_path, )