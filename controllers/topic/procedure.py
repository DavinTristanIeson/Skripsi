
import os

from common.logger import RegisteredLogger, TimeLogger
from common.task.executor import TaskPayload
from common.task.responses import TaskResponseData, TaskStatusEnum
from controllers.topic.builder import BERTopicIndividualModels, BERTopicModelBuilder
from controllers.topic.embedding import bertopic_embedding
from controllers.topic.postprocess import bertopic_post_processing
from controllers.topic.preprocess import bertopic_preprocessing
from controllers.topic.utils import BERTopicColumnIntermediateResult, assert_valid_workspace_for_topic_modeling
from models.config.paths import ProjectPaths
from models.project.cache import ProjectCacheManager

logger = RegisteredLogger().provision("Topic Modeling")

def bertopic_find_topics(
  intermediate: BERTopicColumnIntermediateResult,
):
  from bertopic import BERTopic
  column = intermediate.column
  config = intermediate.config
  documents = intermediate.embedding_documents
  embeddings = intermediate.embeddings
  task = intermediate.task
  model = intermediate.model

  bertopic_path = config.paths.full_path(os.path.join(ProjectPaths.BERTopic(column.name)))

  if os.path.exists(bertopic_path):
    try:
      task.log_pending(f"Loaded cached BERTopic model for \"{column.name}\" from \"{bertopic_path}\".")
      new_model = BERTopic.load(bertopic_path)
      new_model.embedding_model = model.embedding_model
      new_model.umap_model = model.umap_model
      new_model.hdbscan_model = model.hdbscan_model
      task.log_success(f"Loaded cached BERTopic model for \"{column.name}\" from \"{bertopic_path}\".")
      intermediate.model = new_model
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
  intermediate.document_topic_assignments = topics

  task.log_success(f"Saved BERTopic model in \"{bertopic_path}\".")
  model.save(bertopic_path, "safetensors", save_ctfidf=True)


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
  intermediates: list[BERTopicColumnIntermediateResult] = list(map(
    lambda column: BERTopicColumnIntermediateResult.initialize(
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
    df.to_parquet(workspace_path)
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
    result.save_as_json(intermediate.column)

  df.to_parquet(workspace_path)
  task.log_success("Finished discovering topics in Project \"{task.request.project_id}\" (data sourced from {config.source.path})")
  task.success(TaskResponseData.Empty())
