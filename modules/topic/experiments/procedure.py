from dataclasses import dataclass
import datetime
import functools
import queue
import threading
from typing import TYPE_CHECKING, Sequence, cast
from copy import copy

from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.exceptions.files import FileLoadingException
from modules.logger.provisioner import ProvisionedLogger
from modules.task.responses import TaskResponse
from modules.task.manager import TaskManagerProxy
from modules.topic.evaluation.evaluate import evaluate_topics
from modules.topic.experiments.model import BERTopicExperimentResult, BERTopicExperimentTrialResult, BERTopicHyperparameterConstraint
from modules.topic.procedure.base import BERTopicIntermediateState, BERTopicProcedureComponent
from modules.topic.procedure.embedding import BERTopicCacheOnlyEmbeddingProcedureComponent
from modules.topic.procedure.model_builder import BERTopicModelBuilderProcedureComponent, BERTopicWithoutEmbeddingsModelBuilderProcedureComponent
from modules.topic.procedure.postprocess import BERTopicPostprocessProcedureComponent
from modules.topic.procedure.preprocess import BERTopicCacheOnlyPreprocessProcedureComponent, BERTopicDataLoaderProcedureComponent
from modules.topic.procedure.topic_modeling import BERTopicCacheOnlyTopicModelingProcedureComponent, BERTopicExperimentalTopicModelingProcedureComponent

if TYPE_CHECKING:
  from optuna.trial import Trial

logger = ProvisionedLogger().provision("Topic Modeling")

@dataclass
class BERTopicExperimentLab:
  task: TaskManagerProxy
  project_id: str
  column: str
  n_trials: int
  constraint: BERTopicHyperparameterConstraint

  def get_placeholder_task(self, trial_id: int):
    return TaskManagerProxy(
      id=f"Trial {trial_id}",
      # but don't share queue. We discard the contents of queue.
      queue=queue.Queue(),
      # Share stop events
      stop_event=self.task.stop_event,
      response=TaskResponse.Idle("placeholder"),
    )

  def experiment(self, trial: "Trial", shared_state: BERTopicIntermediateState, experiment_result: BERTopicExperimentResult, lock: threading.Lock, evaluation_semaphore: threading.Semaphore):
    cache = shared_state.cache
    column = shared_state.column

    if self.task.stop_event.is_set():
      trial.study.stop()

    candidate = self.constraint.suggest(trial)
    placeholder_task = self.get_placeholder_task(trial.number)
    self.task.log_pending(f"Running a trial for the following hyperparameters: {candidate}")

    # Shallow copy only, don't deep copy.
    state = copy(shared_state)
    state.column = candidate.apply(column, copy=True)

    try:
      procedures: list[BERTopicProcedureComponent] = [
        BERTopicModelBuilderProcedureComponent(state=state, task=placeholder_task),
        BERTopicExperimentalTopicModelingProcedureComponent(state=state, task=placeholder_task),
        BERTopicPostprocessProcedureComponent(state=state, task=placeholder_task, can_save=False),
      ]
      for procedure in procedures:
        procedure.run()
      with evaluation_semaphore:
        evaluation = evaluate_topics(
          documents=cast(Sequence[str], state.documents),
          topics=state.result.topics,
          bertopic_model=state.model,
          document_topic_assignments=state.document_topic_assignments,
        )
    except Exception as e:
      logger.exception(e)
      self.task.log_error(f"Failed to run trial {trial.number + 1} with the following hyperparameters: {candidate} due to the following error: {str(e)}.")
      result = BERTopicExperimentTrialResult(
        evaluation=None,
        candidate=candidate,
        error=str(e),
        trial_number=trial.number + 1,
      )
      with lock:
        experiment_result.trials.append(result)
        cache.bertopic_experiments.save(experiment_result, column.name)
      return -1.0

    self.task.log_success(f"Finished running trial {trial.number + 1} with the following hyperparameters: {candidate} with coherence score of {evaluation.coherence_v:.4f} and diversity of {evaluation.topic_diversity:.4f}.")
    result = BERTopicExperimentTrialResult(
      evaluation=evaluation,
      candidate=candidate,
      error=None,
      trial_number=trial.number + 1,
    )
    with lock:
      experiment_result.trials.append(result)
      cache.bertopic_experiments.save(experiment_result, column.name)

    return evaluation.coherence_v
  
  def evaluate_current(self, shared_state: BERTopicIntermediateState, column: TextualSchemaColumn):
    placeholder_task = self.get_placeholder_task(0)
    state = copy(shared_state)
    more_procedures: list[BERTopicProcedureComponent] = [
      BERTopicWithoutEmbeddingsModelBuilderProcedureComponent(state=state, task=placeholder_task),
      BERTopicCacheOnlyTopicModelingProcedureComponent(state=state, task=placeholder_task),
      BERTopicPostprocessProcedureComponent(state=state, task=placeholder_task, can_save=False),
    ]
    self.task.log_pending("Loading cached documents and topics for topic evaluation.")
    for procedure in more_procedures:
      procedure.run()
    self.task.log_success("Loaded cached documents and topics for topic evaluation.")

    self.task.log_pending("Evaluating the current topics...")
    evaluation = evaluate_topics(
      documents=cast(Sequence[str], state.documents),
      topics=state.result.topics,
      bertopic_model=state.model,
      document_topic_assignments=state.document_topic_assignments,
    )
    self.task.log_success("Finished evaluating the current topics.")

    return evaluation
  
  def run(self):
    shared_state = BERTopicIntermediateState()
    placeholder_task = self.get_placeholder_task(0)

    start_time = datetime.datetime.now()

    shared_procedures: list[BERTopicProcedureComponent] = [
      BERTopicDataLoaderProcedureComponent(state=shared_state, task=placeholder_task, project_id=self.project_id, column=self.column),
      BERTopicCacheOnlyPreprocessProcedureComponent(state=shared_state, task=placeholder_task),
      # load UMAP vectors
      BERTopicCacheOnlyEmbeddingProcedureComponent(state=shared_state, task=placeholder_task),
    ]
    for procedure in shared_procedures:
      procedure.run()
    # Loaded from DataLoaderProcedureComponent
    column = shared_state.column
    cache = shared_state.cache

    try:
      evaluation = cache.topic_evaluations.load(column.name)
    except FileLoadingException as e:
      evaluation = self.evaluate_current(shared_state=shared_state, column=column)

    now = datetime.datetime.now()
    experiment_result = BERTopicExperimentResult(
      trials=[],
      constraint=self.constraint,
      end_at=None,
      last_updated_at=now,
      start_at=now,
      max_trials=self.n_trials,
      evaluation=evaluation,
    )
    self.task.response.data = experiment_result

    experiment = functools.partial(
      self.experiment,
      shared_state=shared_state,
      experiment_result=experiment_result,
      lock=threading.Lock(),
      # Coherence V evaluation creates new processes, so we want to limit it to just two at a time to prevent it from spawning too many processes.
      evaluation_semaphore=threading.Semaphore(2),
    )
    
    import optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(
      experiment,
      n_trials=self.n_trials,
      show_progress_bar=True,
      # Four at a time. HDBSCAN and C-TF-IDF are performant enough for this. Plus HDBSCAN got joblib for parameter caching.
      n_jobs=4,
      catch=Exception,
      gc_after_trial=True,
      callbacks=[]
    )
    experiment_result.end_at = datetime.datetime.now()
    cache.bertopic_experiments.save(experiment_result, column.name)

    end_time = datetime.datetime.now()
    self.task.log_success(f"Experiment has completed successfully (Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}).")

    self.task.success(experiment_result)
  
__all__ = [
  "BERTopicExperimentLab",
]