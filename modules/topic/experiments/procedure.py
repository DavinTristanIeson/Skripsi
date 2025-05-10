from dataclasses import dataclass
import datetime
import functools
import multiprocessing
import threading
from typing import TYPE_CHECKING, Sequence, cast
from copy import copy

from modules.logger.provisioner import ProvisionedLogger
from modules.task.responses import TaskResponse
from modules.task.manager import TaskManagerProxy
from modules.topic.evaluation.evaluate import evaluate_topics
from modules.topic.experiments.model import BERTopicExperimentResult, BERTopicExperimentTrialResult, BERTopicHyperparameterConstraint
from modules.topic.procedure.base import BERTopicIntermediateState, BERTopicProcedureComponent
from modules.topic.procedure.model_builder import BERTopicModelBuilderProcedureComponent
from modules.topic.procedure.postprocess import BERTopicPostprocessProcedureComponent
from modules.topic.procedure.preprocess import BERTopicCacheOnlyPreprocessProcedureComponent, BERTopicDataLoaderProcedureComponent
from modules.topic.procedure.topic_modeling import BERTopicExperimentalTopicModelingProcedureComponent

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

  def experiment(self, trial: "Trial", shared_state: BERTopicIntermediateState, experiment_result: BERTopicExperimentResult, lock: threading.Lock):
    cache = shared_state.cache
    column = shared_state.column

    candidate = self.constraint.suggest(trial)
    placeholder_id = f"Candidate {trial._trial_id}"
    response_queue = multiprocessing.Queue()
    placeholder_task = TaskManagerProxy(
      id=placeholder_id,
      # but don't share queue. We discard the contents of queue.
      queue=response_queue,
      # Share stop events
      stop_event=self.task.stop_event,
      response=TaskResponse.Idle(placeholder_id),
    )
    self.task.log_pending(f"Running a trial for the following hyperparameters: {candidate}")

    # Shallow copy only, don't deep copy.
    state = copy(shared_state)
    state.column = candidate.apply(column)

    try:
      procedures: list[BERTopicProcedureComponent] = [
        BERTopicModelBuilderProcedureComponent(state=state, task=placeholder_task),
        BERTopicExperimentalTopicModelingProcedureComponent(state=state, task=placeholder_task),
        BERTopicPostprocessProcedureComponent(state=state, task=placeholder_task, can_save=False),
      ]
      for procedure in procedures:
        procedure.run()
      evaluation = evaluate_topics(cast(Sequence[str], state.documents), state.result.topics, state.model)
    except Exception as e:
      self.task.log_error(f"Failed to run a trial for the following hyperparameters: {candidate} due to the following error: {str(e)}.")
      result = BERTopicExperimentTrialResult(
        evaluation=None,
        candidate=candidate,
        error=str(e),
      )
      with lock:
        experiment_result.trials.append(result)
        cache.bertopic_experiments.save(experiment_result, column.name)
        raise e

    self.task.log_success(f"Finished running a trial for the following hyperparameters: {candidate} with coherence score of {evaluation.coherence_v} and diversity of {evaluation.topic_diversity}.")
    result = BERTopicExperimentTrialResult(
      evaluation=evaluation,
      candidate=candidate,
      error=None
    )
    with lock:
      experiment_result.trials.append(result)
      cache.bertopic_experiments.save(experiment_result, column.name)

    return evaluation.coherence_v
  
  def run(self):
    shared_state = BERTopicIntermediateState()
    shared_procedures: list[BERTopicProcedureComponent] = [
      BERTopicDataLoaderProcedureComponent(state=shared_state, task=self.task, project_id=self.project_id, column=self.column),
      BERTopicCacheOnlyPreprocessProcedureComponent(state=shared_state, task=self.task),
    ]
    for procedure in shared_procedures:
      procedure.run()
    # Loaded from DataLoaderProcedureComponent
    column = shared_state.column

    now = datetime.datetime.now()
    experiment_result = BERTopicExperimentResult(
      trials=[],
      end_at=None,
      last_updated_at=now,
      start_at=now,
    )

    experiment = functools.partial(
      self.experiment,
      shared_state=shared_state,
      experiment_result=experiment_result,
      lock=threading.Lock(),
    )
    
    import optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(
      experiment,
      n_trials=self.n_trials,
      show_progress_bar=True,
      # Optuna's n_jobs uses multithreading, not multiprocessing so this is save to use.
      n_jobs=-1,
      catch=Exception,
      gc_after_trial=True,
      callbacks=[]
    )
    experiment_result.end_at = datetime.datetime.now()
    shared_state.cache.bertopic_experiments.save(experiment_result, column.name)

    self.task.success(experiment_result)
  
__all__ = [
  "BERTopicExperimentLab",
]