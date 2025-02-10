
import os

import pandas as pd
from common.task.executor import TaskPayload
from controllers.topic.embedding import bertopic_embedding
from controllers.topic.preprocess import bertopic_preprocessing
from controllers.topic.utils import BERTopicColumnIntermediateResult, assert_valid_workspace_for_topic_modeling
from models.config.paths import ProjectPaths
from models.project.cache import ProjectCacheManager

def run_topic_modeling_procedure(task: TaskPayload):
  cache = ProjectCacheManager().get(task.request.project_id)
  config = cache.config

  workspace_path = config.paths.full_path(ProjectPaths.Workspace)
  task.progress(f"Loading cached dataset from \"{workspace_path}\"...")
  df = cache.load_workspace()
  task.progress(f"Loaded cached dataset from \"{workspace_path}\"...")

  assert_valid_workspace_for_topic_modeling(
    df=df,
    config=config,
    task=task,
  )

  intermediates: list[BERTopicColumnIntermediateResult] = map(lambda column: BERTopicColumnIntermediateResult(
    column=column,
    config=config,
    documents=None,
    mask=None,
    embeddings=None,
    embedding_model=None,
    task=task
  ), textual_columns) # type: ignore

  textual_columns = config.data_schema.textual()
  preprocess_count = 0
  for idx, intermediate in enumerate(intermediates):
    if not intermediate.column.preprocess_column.name in df.columns:
      preprocess_count += 1

    bertopic_preprocessing(
      df=df,
      intermediate=intermediate
    )
  
  if preprocess_count > 0:  
    df.to_parquet(workspace_path)
    task.progress(f"Saved preprocessed documents to {workspace_path}.")

  for idx, intermediate in enumerate(intermediates):
    embedding_result = bertopic_embedding(intermediate)

  for idx, intermediate in enumerate(intermediate):
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
