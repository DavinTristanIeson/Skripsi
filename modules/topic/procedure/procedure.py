
import os

import numpy as np

from modules.logger import ProvisionedLogger, TimeLogger
from modules.project.cache import ProjectCacheManager
from modules.task import TaskPayload, TaskResponseData
from modules.project.paths import ProjectPaths

from ..bertopic_ext import BERTopicIndividualModels, BERTopicModelBuilder
from .embedding import bertopic_embedding
from .postprocess import bertopic_post_processing
from .preprocess import bertopic_preprocessing
from .utils import _BERTopicColumnIntermediateResult, assert_valid_workspace_for_topic_modeling

logger = ProvisionedLogger().provision("Topic Modeling")

def bertopic_find_topics(
  intermediate: _BERTopicColumnIntermediateResult,
):
  from bertopic import BERTopic
  column = intermediate.column
  config = intermediate.config
  documents = list(intermediate.documents)
  embeddings = intermediate.embeddings
  task = intermediate.task
  model = intermediate.model

  cache = ProjectCacheManager().get(project_id=config.project_id)

  bertopic_path = config.paths.full_path(os.path.join(ProjectPaths.BERTopic(column.name)))

  if os.path.exists(bertopic_path):
    try:
      task.log_pending(f"Loaded cached BERTopic model for \"{column.name}\" from \"{bertopic_path}\".")
      
      new_model = BERTopic.load(bertopic_path, embedding_model=model.embedding_model)
      new_model.umap_model = model.umap_model
      new_model.hdbscan_model = model.hdbscan_model
      task.log_success(f"Loaded cached BERTopic model for \"{column.name}\" from \"{bertopic_path}\".")

      intermediate.model = new_model
      intermediate.document_topic_assignments = np.array(new_model.topics_, dtype=np.int32)

      if not new_model.topics_ or len(new_model.topics_) != len(intermediate.documents):
        # Can't use cached model.
        task.log_error(f"Cached BERTopic model in {bertopic_path} is not synchronized with current dataset. Re-fitting BERTopic model again.")
      else:
        # We can just return since we're already using the cached model
        return
    except Exception as e:
      task.log_error(f"Failed to load cached BERTopic model from {bertopic_path}. Re-fitting BERTopic model again.")
      logger.error(e)

  task.log_pending(f"Starting the topic modeling process for \"{column.name}\".")

  with TimeLogger("Topic Modeling", "Performing Topic Modeling", report_start=True):
    topics, probs = model.fit_transform(documents, embeddings)

  task.log_success(f"Finished the topic modeling process for {column.name}. Performing additional post-processing for the discovered topics.")
  logger.info(f"Topics of {intermediate.column.name}: {model.topic_labels_}. ")

  if column.topic_modeling.no_outliers:
    topics = model.reduce_outliers(documents, topics, strategy="embeddings", embeddings=embeddings)
    if column.topic_modeling.represent_outliers:
      model.update_topics(intermediate.embedding_documents, topics=topics)

  intermediate.model = model
  intermediate.document_topic_assignments = np.array(topics, dtype=np.int32)

  task.log_success(f"Saved BERTopic model in \"{bertopic_path}\".")
  cache.save_bertopic(model, column.name)


def bertopic_topic_modeling(task: TaskPayload):
  cache = ProjectCacheManager().get(task.request.project_id)
  config = cache.config

  # LOAD WORKSPACE
  workspace_path = config.paths.full_path(ProjectPaths.Workspace)
  task.log_pending(f"Loading cached dataset from \"{workspace_path}\"...")
  df = cache.load_workspace()
  task.log_success(f"Loaded cached dataset from \"{workspace_path}\"...")

  assert_valid_workspace_for_topic_modeling(
    df=df,
    config=config,
    task=task,
  )

  textual_columns = config.data_schema.textual()
  intermediates: list[_BERTopicColumnIntermediateResult] = list(map(
    lambda column: _BERTopicColumnIntermediateResult.initialize(
      column=column,
      config=config,
      task=task
    ),
    textual_columns
  ))

  preprocess_count = 0
  # DATASET PREPROCESSING
  for intermediate in intermediates:
    if not intermediate.column.preprocess_column.name in df.columns:
      preprocess_count += 1

    bertopic_preprocessing(
      df=df,
      intermediate=intermediate
    )
  
  if preprocess_count > 0:  
    config.save_workspace(df)
    task.log_success(f"Saved preprocessed documents to {workspace_path}.")

  # MODEL CREATION
  for intermediate in intermediates:
    intermediate.model = BERTopicModelBuilder(
      project_id=config.project_id,
      column=intermediate.column,
      corpus_size=len(intermediate.documents)
    ).build()

  # DOCUMENT EMBEDDING
  for intermediate in intermediates:
    embedding_model = BERTopicIndividualModels.cast(intermediate.model).embedding_model
    bertopic_embedding(
      embedding_model,
      intermediate
    )

  for intermediate in intermediates:
    # TOPIC MODELING
    bertopic_find_topics(intermediate)
    
    # TOPIC POST PREPROCESSING
    result = bertopic_post_processing(df, intermediate)
    topics_path = config.paths.allocate_path(ProjectPaths.Topics(intermediate.column.name))
    task.log_success(f"Saving topics of \"{intermediate.column.name}\" in \"{topics_path}\"")
    result.save_as_json(intermediate.column.name)

  config.save_workspace(df)
  task.log_success(f"Finished discovering topics in Project \"{task.request.project_id}\" (data sourced from {config.source.path})")
  task.success(TaskResponseData.Empty())

  cache.workspaces.clear()

__all__ = [
  "bertopic_topic_modeling",
]