from dataclasses import dataclass
import threading
from typing import cast

from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.project.cache import ProjectCache
from modules.project.lock import ProjectFileLockManager
from modules.project.paths import ProjectPathManager, ProjectPaths
from modules.task.proxy import TaskManagerProxy
from modules.topic.bertopic_ext.dimensionality_reduction import BERTopicCachedUMAP
from modules.topic.evaluation.evaluate import evaluate_topics
from modules.topic.exceptions import MissingCachedTopicModelingResult, UnsyncedDocumentVectorsException
from modules.topic.experiments.model import BERTopicHyperparameterConstraint
from modules.topic.experiments.procedure import BERTopicExperimentLab
from modules.topic.procedure.procedure import BERTopicProcedureFacade

@dataclass
class TopicModelingTaskRequest:
  project_id: str
  column: str

  @property
  def task_id(self):
    # To enable sequential runs. The ID makes all topic modeling jobs for the same project ID the same, while misfire_grace_time prevents the jobs from being canceled
    # This is necessary to avoid data races for the same project.
    # https://stackoverflow.com/questions/65690003/how-to-manage-a-task-queue-using-apscheduler
    return f"{self.project_id}__{self.column}__topic-modeling"
  
@dataclass
class EvaluateTopicModelResultTaskRequest:
  project_id: str
  column: str

  @property
  def task_id(self):
    return f"{self.project_id}__{self.column}__evaluate-topic-model-result"

@dataclass
class BERTopicExperimentTaskRequest:
  project_id: str
  column: str
  n_trials: int
  constraint: BERTopicHyperparameterConstraint

  @property
  def task_id(self):
    return f"{self.project_id}__{self.column}__bertopic-experiment"

def topic_evaluation_task_inner(proxy: TaskManagerProxy, request: EvaluateTopicModelResultTaskRequest):
  # We're only using the writer 
  cache = ProjectCache(
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
  raw_document_topic_assignments = df[column.topic_column.name]
  document_topic_assignments = raw_document_topic_assignments[mask]

  tm_result = cache.topics.load(column.name)
  bertopic_model = cache.bertopic_models.load(column.name)
  umap_model = BERTopicCachedUMAP(
    project_id=request.project_id,
    column=column,
    low_memory=True,
  )
  cached_umap_vectors = umap_model.load_cached_embeddings()
  if cached_umap_vectors is None:
    raise MissingCachedTopicModelingResult(
      type="UMAP vectors",
      column=column.name
    )
  if len(cached_umap_vectors) != len(document_topic_assignments):
    raise UnsyncedDocumentVectorsException(
      type="UMAP vectors",
      expected_rows=len(document_topic_assignments),
      observed_rows=len(cached_umap_vectors),
      column=column.name,
    )
  proxy.log_success(f"Successfully loaded cached documents and topics for {column.name}.")

  proxy.log_pending(f"Evaluating the topics...")
  result = evaluate_topics(
    bertopic_model=bertopic_model,
    documents=documents,
    topics=tm_result.topics,
    document_topic_assignments=document_topic_assignments,
    umap_vectors=cached_umap_vectors,
  )
  proxy.log_success("Finished evaluating the topics.")
  proxy.success(result)
  cache.topic_evaluations.save(result, column.name)

# PROCESS CONTEXT WARN
def topic_evaluation_task(proxy: TaskManagerProxy, request: EvaluateTopicModelResultTaskRequest):
  paths = ProjectPathManager(project_id=request.project_id)
  log_path = paths.allocate_path(ProjectPaths.TopicEvaluationLogs(request.column))
  with proxy.context(log_file=log_path):
    file_lock = ProjectFileLockManager().lock_column(
      project_id=request.project_id,
      column=request.column,
      wait=False
    )
    with file_lock:
      topic_evaluation_task_inner(proxy, request)

def topic_modeling_task(proxy: TaskManagerProxy, request: TopicModelingTaskRequest):
  paths = ProjectPathManager(project_id=request.project_id)
  log_path = paths.allocate_path(ProjectPaths.TopicModelingLogs(request.column))
  with proxy.context(log_file=log_path):
    file_lock = ProjectFileLockManager().lock_column(
      project_id=request.project_id,
      column=request.column,
      wait=False,
    )
    with file_lock:
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
    file_lock = ProjectFileLockManager().lock_column(
      project_id=request.project_id,
      column=request.column,
      wait=False
    )
    with file_lock:
      study = BERTopicExperimentLab(
        task=proxy,
        column=request.column,
        project_id=request.project_id,
        constraint=request.constraint,
        n_trials=request.n_trials,
      )
      study.run()