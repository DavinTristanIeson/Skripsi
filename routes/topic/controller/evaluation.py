import functools
import http
from typing import cast

from modules.task.convenience import AlternativeTaskResponse, get_task_result_or_else
from typing import cast
from modules.topic.experiments.model import BERTopicExperimentResult, BERTopicHyperparameterCandidate
from routes.dependencies.project import ProjectCacheDependency
from modules.api.wrapper import ApiError, ApiResult
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.exceptions.files import FileLoadingException
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache
from modules.task.responses import TaskResponse
from modules.task.manager import TaskConflictResolutionBehavior, TaskManager
from modules.topic.evaluation.model import TopicEvaluationResult
from routes.topic.controller.tasks import BERTopicExperimentTaskRequest, EvaluateTopicModelResultTaskRequest, topic_evaluation_task, topic_model_experiment_task
from routes.topic.controller.topic_model import start_topic_modeling
from routes.topic.model import StartTopicModelingSchema, TopicModelExperimentSchema

logger = ProvisionedLogger().provision("Topic Controller")

  
def perform_topic_model_evaluation(cache: ProjectCacheDependency, column: TextualSchemaColumn):
  request = EvaluateTopicModelResultTaskRequest(
    project_id=cache.config.project_id,
    column=column.name,
  )

  store = TaskManager()
  store.add_task(
    task_id=request.task_id,
    task=topic_evaluation_task,
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

  alternative_response = functools.partial(
    __get_topic_model_evaluation_result_alternative_response,
    cache=cache,
    column=column.name,
  )
  task_result = get_task_result_or_else(
    task_id=request.task_id,
    alternative_response=alternative_response,
  )
  if task_result is None:
    raise ApiError(f"Topic evaluation has not been started for \"{column.name}\" in project \"{config.metadata.name}\".", http.HTTPStatus.BAD_REQUEST)
  return task_result

def perform_topic_model_experiment(cache: ProjectCache, column: TextualSchemaColumn, body: TopicModelExperimentSchema):
  store = TaskManager()
  request = BERTopicExperimentTaskRequest(
    project_id=cache.config.project_id,
    column=column.name,
    constraint=body.constraint,
    n_trials=body.n_trials,
  )
  store.add_task(
    task_id=request.task_id,
    task=topic_model_experiment_task,
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
  task_result = get_task_result_or_else(
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

  column = candidate.apply(column, copy=False)
  
  config.save_to_json()
  cache.config_cache.invalidate()
  config.paths.cleanup_topic_modeling(column.name)
  TaskManager().invalidate(prefix=f"{config.project_id}__{column.name}", clear=True)

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