
import os

import pandas as pd
from common.logger import TimeLogger
from common.task.executor import TaskPayload
from controllers.topic.embedding import bertopic_embedding
from controllers.topic.preprocess import bertopic_load_workspace, bertopic_preprocessing
from controllers.topic.utils import BERTopicColumnIntermediateResult, assert_valid_workspace_for_topic_modeling
from models.config.paths import ProjectPaths
from models.project.cache import ProjectCacheManager

def run_topic_modeling_procedure(task: TaskPayload):
  df = bertopic_load_workspace(task)

  cache = ProjectCacheManager().get(task.request.project_id)
  config = cache.config
  workspace_path = config.paths.full_path(ProjectPaths.Workspace)

  assert_valid_workspace_for_topic_modeling(
    df=df,
    config=config,
    task=task,
  )

  intermediates: list[BERTopicColumnIntermediateResult] = []
  textual_columns = config.data_schema.textual()
  preprocess_count = 0
  for idx, column in enumerate(textual_columns):
    if not column.preprocess_column.name in df.columns:
      preprocess_count += 1

    intermediate = bertopic_preprocessing(
      df=df,
      task=task,
      column=column,
      index=idx,
      total=len(textual_columns),
    )
    intermediates.append(intermediate)
  
  if preprocess_count > 0:  
    df.to_parquet(workspace_path)
    task.progress(f"Saved preprocessed documents to {workspace_path}.")

  for idx, intermediate in enumerate(intermediates):
    intermediate.embeddings = bertopic_embedding(
      documents=intermediate.documents,
      column=intermediate.column,
      config=config,
      task=task,
      index=idx,
      total=len(intermediates)
    )

  textcolumns = config.data_schema.textual()
  for colidx, column in enumerate(textcolumns):
    column_progress = f"({colidx + 1} / {len(textcolumns)})"
    column_data: pd.Series = df[column.preprocess_column]
    mask = column_data.str.len() != 0
    documents: list[str] = list(column_data[mask])

    if len(documents) == 0:
      raise ValueError(f"{column.name} does not contain any valid documents after the preprocessing step. Either change the preprocessing configuration of {column.name} to be more lax (e.g: lower the min word frequency, min document length), or set the type of this column to Unique.")

    task.check_stop()
    task.progress(
      progress=steps.advance(),
      message=f"Transforming documents of {column.name} into document embeddings {column_progress}"
    )

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
  df.to_parquet(workspace_path)
  task.success(IPCResponseData.Empty(), message=f"Finished discovering topics in Project \"{task.request.project_id}\" (data sourced from {config.source.path})")
