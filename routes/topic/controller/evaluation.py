import functools
import http
from typing import Sequence, cast
from controllers.project import ProjectCacheDependency
from modules.api.wrapper import ApiError
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
from routes.topic.model import BERTopicExperimentTaskRequest, EvaluateTopicModelResultTaskRequest

logger = ProvisionedLogger().provision("Topic Controller")

def _perform_topic_model_evaluation_task(payload: EvaluateTopicModelResultTaskRequest):
  taskstore = TaskStorage()
  with taskstore.proxy_context(payload.task_id) as proxy:
    cache = ProjectCacheManager().get(payload.project_id)
  
    config = cache.config
    df = cache.workspaces.load()
    column = cast(TextualSchemaColumn, config.data_schema.assert_of_type(payload.column, [SchemaColumnTypeEnum.Textual]))

    column.assert_internal_columns(df, with_preprocess=True, with_topics=False)
    
    proxy.log_pending(f"Loading cached documents and topics for {column.name}")
    raw_documents = df[column.preprocess_column]
    mask = raw_documents.notna() & raw_documents != ''
    raw_documents = raw_documents[mask]

    tm_result = cache.topics.load(column.name)
    bertopic_model = cache.bertopic_models.load(column.name)
    proxy.log_success(f"Successfully loaded cached documents and topics for {column.name}.")

    proxy.log_pending(f"Evaluating the topics...")
    result = evaluate_topics(
      bertopic_model=bertopic_model,
      raw_documents=cast(Sequence[str], raw_documents),
      topics=tm_result.topics,
    )
    proxy.log_success("Finished evaluating the topics.")
    proxy.success(result)
    cache.topic_evaluations.save(result, column.name)
  
def perform_topic_model_evaluation(request: EvaluateTopicModelResultTaskRequest):
  store = TaskStorage()
  store.add_task(
    scheduler=scheduler,
    task_id=request.task_id,
    task=_perform_topic_model_evaluation_task,
    args=[request],
    idle_message=f"Beginning the evaluation of the topic modeling results of \"{request.column}\".",
    conflict_resolution=TaskConflictResolutionBehavior.Ignore,
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

def perform_topic_model_experiment(request: BERTopicExperimentTaskRequest):
  store = TaskStorage()
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

def check_topic_model_experiment_status(cache: ProjectCache, request: BERTopicExperimentTaskRequest)->TaskResponse[TopicEvaluationResult]:
  store = TaskStorage()

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
    raise ApiError(f"Topic evaluation has not been started for \"{request.column}\" in project \"{cache.config.metadata.name}\".", http.HTTPStatus.BAD_REQUEST)
  return task_result
