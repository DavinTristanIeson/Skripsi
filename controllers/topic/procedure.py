
import os

from common.logger import RegisteredLogger
from common.task.executor import TaskPayload
from common.task.responses import TaskResponse, TaskResponseData
from controllers.topic.embedding import bertopic_embedding
from controllers.topic.modeling import bertopic_topic_modeling
from controllers.topic.preprocess import bertopic_preprocessing
from controllers.topic.utils import BERTopicColumnIntermediateResult, assert_valid_workspace_for_topic_modeling
from models.config.paths import ProjectPaths
from models.project.cache import ProjectCacheManager

logger = RegisteredLogger().provision("Topic Modeling")

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

  textual_columns = config.data_schema.textual()
  intermediates: list[BERTopicColumnIntermediateResult] = list(map(
    lambda column: BERTopicColumnIntermediateResult.initialize(
      column=column,
      config=config,
      task=task
    ),
    textual_columns
  ))

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
    bertopic_embedding(intermediate)

  for idx, intermediate in enumerate(intermediates):
    bertopic_topic_modeling(intermediate)
    model = intermediate.model
    logger.info(f"Topics of {intermediate.column.name}: {model.topic_labels_}. ")

    bertopic_path = config.paths.allocate_path(os.path.join(ProjectPaths.BERTopic(intermediate.column.name)))
    task.progress(f"Saving BERTopic model in \"{bertopic_path}\".")
    model.save(bertopic_path, "safetensors", save_ctfidf=True)
      
  df.to_parquet(workspace_path)
  task.success(TaskResponseData.Empty(), message=f"Finished discovering topics in Project \"{task.request.project_id}\" (data sourced from {config.source.path})")
