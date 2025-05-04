from dataclasses import dataclass, field
import datetime
import functools
import threading
from typing import TYPE_CHECKING, Sequence, cast
from copy import copy

from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCacheManager
from modules.task.responses import TaskResponse
from modules.task.storage import TaskStorageProxy
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
  task: TaskStorageProxy
  project_id: str
  column: str
  n_trials: int
  constraint: BERTopicHyperparameterConstraint

  def experiment(self, trial: "Trial", shared_state: BERTopicIntermediateState, column: TextualSchemaColumn, experiment_result: BERTopicExperimentResult):
    cache = ProjectCacheManager().get(self.project_id)

    candidate = self.constraint.suggest(trial)
    placeholder_id = f"Candidate {trial._trial_id}"
    placeholder_task = TaskStorageProxy(
      id=placeholder_id,
      stop_event=threading.Event(),
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
        topic_modeling_config=state.column.topic_modeling,
        error=str(e),
      )
      experiment_result.trials.append(result)
      cache.bertopic_experiments.save(experiment_result, column.name)
      raise e

    self.task.log_success(f"Finished running a trial for the following hyperparameters: {candidate} with coherence score of {evaluation.coherence_v} and diversity of {evaluation.topic_diversity}.")
    result = BERTopicExperimentTrialResult(
      evaluation=evaluation,
      topic_modeling_config=state.column.topic_modeling,
      error=None
    )
    experiment_result.trials.append(result)
    cache.bertopic_experiments.save(experiment_result, column.name)

    return evaluation.coherence_v
  
  def run(self):
    cache = ProjectCacheManager().get(self.project_id)
    config = cache.config
    column = cast(TextualSchemaColumn, config.data_schema.assert_of_type(self.column, [SchemaColumnTypeEnum.Textual]))
    
    shared_state = BERTopicIntermediateState()
    shared_procedures: list[BERTopicProcedureComponent] = [
      BERTopicDataLoaderProcedureComponent(state=shared_state, task=self.task),
      BERTopicCacheOnlyPreprocessProcedureComponent(state=shared_state, task=self.task),
    ]
    for procedure in shared_procedures:
      procedure.run()

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
      column=column,
      experiment_result=experiment_result
    )
    
    import optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(
      experiment,
      n_trials=self.n_trials,
      show_progress_bar=True,
      # Optuna's n_jobs uses multithreading, not multiprocessing
      n_jobs=-1,
      catch=Exception,
      gc_after_trial=True,
      callbacks=[]
    )
    experiment_result.end_at = datetime.datetime.now()
    cache.bertopic_experiments.save(experiment_result, column.name)

    self.task.success(experiment_result)
  
__all__ = [
  "BERTopicExperimentLab",
]