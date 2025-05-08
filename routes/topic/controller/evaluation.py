import functools
import http

from modules.task.convenience import AlternativeTaskResponse, get_task_result_or_else
from routes.dependencies.project import ProjectCacheDependency
from modules.api.wrapper import ApiError
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.exceptions.files import FileLoadingException
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache
from modules.task.responses import TaskResponse
from modules.task.manager import TaskConflictResolutionBehavior, TaskManager
from modules.topic.evaluation.model import TopicEvaluationResult
from routes.topic.controller.tasks import topic_evaluation_task, topic_model_experiment_task
from routes.topic.model import BERTopicExperimentTaskRequest, EvaluateTopicModelResultTaskRequest, TopicModelExperimentSchema

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
    args=[request, store.proxy(request.task_id)],
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

def check_topic_model_experiment_status(cache: ProjectCache, column: TextualSchemaColumn, body: TopicModelExperimentSchema)->TaskResponse[TopicEvaluationResult]:
  store = TaskManager()
  request = BERTopicExperimentTaskRequest(
    project_id=cache.config.project_id,
    column=column.name,
    constraint=body.constraint,
    n_trials=body.n_trials,
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
    raise ApiError(f"Topic evaluation has not been started for \"{request.column}\" in project \"{cache.config.metadata.name}\".", http.HTTPStatus.BAD_REQUEST)
  return task_result
