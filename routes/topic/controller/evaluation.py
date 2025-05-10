import functools
import http
from typing import Sequence, cast
from modules.config.config import Config
from modules.topic.experiments.model import BERTopicExperimentResult, BERTopicHyperparameterCandidate
from routes.dependencies.project import ProjectCacheDependency
from modules.api.wrapper import ApiError, ApiResult
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.exceptions.files import FileLoadingException
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache, ProjectCacheManager
from modules.task.engine import scheduler
from modules.task.responses import TaskResponse, TaskStatusEnum
from modules.task.storage import AlternativeTaskResponse, TaskConflictResolutionBehavior, TaskStorage
from modules.topic.evaluation.evaluate import evaluate_topics
from modules.topic.evaluation.model import TopicEvaluationResult
from modules.topic.experiments.procedure import BERTopicExperimentLab
from routes.topic.controller.task import start_topic_modeling
from routes.topic.model import BERTopicExperimentTaskRequest, EvaluateTopicModelResultTaskRequest, StartTopicModelingSchema, TopicModelExperimentSchema

logger = ProvisionedLogger().provision("Topic Controller")

def _perform_topic_model_evaluation_task(payload: EvaluateTopicModelResultTaskRequest):
  taskstore = TaskStorage()
  with taskstore.proxy_context(payload.task_id) as proxy:
    cache = ProjectCacheManager().get(payload.project_id)
  
    config = cache.config
    column = cast(TextualSchemaColumn, config.data_schema.assert_of_type(payload.column, [SchemaColumnTypeEnum.Textual]))

    proxy.log_pending(f"Loading cached documents and topics for \"{column.name}\"")
    df = cache.workspaces.load()
    column.assert_internal_columns(df, with_preprocess=True, with_topics=False)
    
    raw_documents = df[column.preprocess_column.name]
    mask = raw_documents.notna() & (raw_documents != '')
    documents: list[str] = raw_documents[mask].to_list()

    tm_result = cache.topics.load(column.name)
    bertopic_model = cache.bertopic_models.load(column.name)
    proxy.log_success(f"Successfully loaded cached documents and topics for {column.name}.")

    proxy.log_pending(f"Evaluating the topics...")
    result = evaluate_topics(
      bertopic_model=bertopic_model,
      raw_documents=documents,
      document_topic_assignments=df[column.topic_column.name],
      topics=tm_result.topics,
    )
    proxy.log_success("Finished evaluating the topics.")
    proxy.success(result)
    cache.topic_evaluations.save(result, column.name)
  
def perform_topic_model_evaluation(cache: ProjectCacheDependency, column: TextualSchemaColumn):
  request = EvaluateTopicModelResultTaskRequest(
    project_id=cache.config.project_id,
    column=column.name,
  )

  store = TaskStorage()
  store.add_task(
    scheduler=scheduler,
    task_id=request.task_id,
    task=_perform_topic_model_evaluation_task,
    args=[request],
    idle_message=f"Beginning the evaluation of the topic modeling results of \"{request.column}\".",
    conflict_resolution=TaskConflictResolutionBehavior.Cancel,
  )

def __get_topic_model_evaluation_result_alternative_response(cache: ProjectCache, column: str):
  try:
    result = cache.topic_evaluations.load(column)
  except FileLoadingException:
    return None
  return AlternativeTaskResponse(
    data=result,
    message="The topics have already been evaluated before. Feel free to explore the discovered topics."
  )

def check_topic_model_evaluation_status(cache: ProjectCacheDependency, column: TextualSchemaColumn)->TaskResponse[TopicEvaluationResult]:
  config = cache.config
  request = EvaluateTopicModelResultTaskRequest(
    project_id=cache.config.project_id,
    column=column.name,
  )

  store = TaskStorage()
  alternative_response = functools.partial(
    __get_topic_model_evaluation_result_alternative_response,
    cache=cache,
    column=column.name,
  )
  task_result = store.get_task_result(
    task_id=request.task_id,
    alternative_response=alternative_response,
  )
  if task_result is None:
    raise ApiError(f"Topic evaluation has not been started for \"{column.name}\" in project \"{config.metadata.name}\".", http.HTTPStatus.BAD_REQUEST)
  return task_result


def _topic_modeling_experimentation_task(payload: BERTopicExperimentTaskRequest):
  taskstore = TaskStorage()
  with taskstore.proxy_context(payload.task_id) as proxy:
    study = BERTopicExperimentLab(
      task=proxy,
      column=payload.column,
      project_id=payload.project_id,
      constraint=payload.constraint,
      n_trials=payload.n_trials,
    )
    study.run()

def perform_topic_model_experiment(cache: ProjectCache, column: TextualSchemaColumn, body: TopicModelExperimentSchema):
  store = TaskStorage()
  request = BERTopicExperimentTaskRequest(
    project_id=cache.config.project_id,
    column=column.name,
    constraint=body.constraint,
    n_trials=body.n_trials,
  )
  store.add_task(
    scheduler=scheduler,
    task_id=request.task_id,
    task=_topic_modeling_experimentation_task,
    args=[request],
    idle_message=f"Beginning the experimentation of the hyperparameters of \"{request.column}\".",
    conflict_resolution=TaskConflictResolutionBehavior.Ignore
  )

def __get_topic_model_experiment_result_alternative_response(cache: ProjectCache, column: str):
  try:
    result = cache.bertopic_experiments.load(column)
  except FileLoadingException:
    return None
  if result.end_at is not None:
    return AlternativeTaskResponse(
      data=result,
      message=f"An experiment had been performed on {column} to find the optimal hyperparameters on {result.end_at.strftime('%Y-%m-%d %H:%M:%S')}."
    )
  else:
    return AlternativeTaskResponse(
      data=result,
      message=f"There had been an experiment on {column} to find the optimal hyperparameters starting from {result.start_at.strftime('%Y-%m-%d %H:%M:%S')}, but that experiment did not finish successfully."
    )

def check_topic_model_experiment_status(cache: ProjectCache, column: TextualSchemaColumn)->TaskResponse[BERTopicExperimentResult]:
  store = TaskStorage()
  request = BERTopicExperimentTaskRequest(
    project_id=cache.config.project_id,
    column=column.name,
    constraint=None, # type: ignore
    n_trials=0,
  )

  alternative_response = functools.partial(
    __get_topic_model_experiment_result_alternative_response,
    cache=cache,
    column=request.column,
  )
  task_result = store.get_task_result(
    task_id=request.task_id,
    alternative_response=alternative_response,
  )
  if task_result is None:
    raise ApiError(f"Topic experiment has not been started for \"{request.column}\" in project \"{cache.config.metadata.name}\".", http.HTTPStatus.BAD_REQUEST)
  return task_result

def apply_topic_model_hyperparameter(
  cache: ProjectCache,
  column_name: str,
  candidate: BERTopicHyperparameterCandidate
):
  config = cache.config.model_copy()
  column = cast(
    TextualSchemaColumn,
    config.data_schema.assert_of_type(
      column_name, [SchemaColumnTypeEnum.Textual]
    )
  )

  if candidate.min_topic_size is not None:
    column.topic_modeling.min_topic_size = candidate.min_topic_size
  if candidate.max_topics is not None:
    column.topic_modeling.max_topics = candidate.max_topics
  if candidate.topic_confidence_threshold is not None:
    column.topic_modeling.topic_confidence_threshold = candidate.topic_confidence_threshold
  
  config.save_to_json()
  cache.config_cache.invalidate()
  config.paths.cleanup_topic_modeling(column.name)
  TaskStorage().invalidate(prefix=f"{config.project_id}__{column.name}", clear=True)

  start_topic_modeling(
    cache=cache,
    column=column,
    options=StartTopicModelingSchema(
      use_cached_document_vectors=True,
      use_cached_umap_vectors=True,
      use_preprocessed_documents=True,
    )
  )

  return ApiResult(data=None, message=f"The topic modeling algorithm will soon be applied to \"{column.name}\" with the specified hyperparameters. Please wait for a few seconds (or minutes depending on the size of your dataset) for the algorithm to complete.")