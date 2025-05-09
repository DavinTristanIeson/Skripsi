import multiprocessing
from multiprocessing.synchronize import Event
import threading
from typing import cast

from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.project.cache import ProjectCache
from modules.project.paths import ProjectPathManager, ProjectPaths
from modules.task.proxy import TaskManagerProxy
from modules.topic.evaluation.evaluate import evaluate_topics
from modules.topic.experiments.procedure import BERTopicExperimentLab
from modules.topic.procedure.procedure import BERTopicProcedureFacade
from routes.topic.model import BERTopicExperimentTaskRequest, EvaluateTopicModelResultTaskRequest, TopicModelingTaskRequest


# PROCESS CONTEXT WARN
def topic_evaluation_task(proxy: TaskManagerProxy, request: EvaluateTopicModelResultTaskRequest):
  paths = ProjectPathManager(project_id=request.project_id)
  log_path = paths.allocate_path(ProjectPaths.TopicEvaluationLogs(request.column))
  with proxy.context(log_file=log_path):
    # We're only using the writer 
    cache = ProjectCache(
      lock=threading.RLock(),
      project_id=request.project_id
    )
  
    config = cache.config
    column = cast(TextualSchemaColumn, config.data_schema.assert_of_type(request.column, [SchemaColumnTypeEnum.Textual]))

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
      topics=tm_result.topics,
    )
    proxy.log_success("Finished evaluating the topics.")
    proxy.success(result)
    cache.topic_evaluations.save(result, column.name)
  
def topic_modeling_task(proxy: TaskManagerProxy, request: TopicModelingTaskRequest):
  paths = ProjectPathManager(project_id=request.project_id)
  log_path = paths.allocate_path(ProjectPaths.TopicModelingLogs(request.column))
  with proxy.context(log_file=log_path):
    facade = BERTopicProcedureFacade(
      task=proxy,
      column=request.column,
      project_id=request.project_id
    )
    facade.run()

def topic_model_experiment_task(proxy: TaskManagerProxy, request: BERTopicExperimentTaskRequest):
  paths = ProjectPathManager(project_id=request.project_id)
  log_path = paths.allocate_path(ProjectPaths.TopicModelExperimentLogs(request.column))
  with proxy.context(log_file=log_path):
    study = BERTopicExperimentLab(
      task=proxy,
      column=request.column,
      project_id=request.project_id,
      constraint=request.constraint,
      n_trials=request.n_trials,
    )
    study.run()