from dataclasses import dataclass
import datetime
import threading
from typing import Sequence, cast
from copy import copy

from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCacheManager
from modules.project.paths import ProjectPaths
from modules.storage.userdata.filesystem import UserDataStorageController
from modules.storage.userdata.resource import UserDataResource, UserDataSchema
from modules.task.storage import TaskStorageProxy
from modules.topic.evaluation.evaluate import evaluate_topics
from modules.topic.experiments.model import BERTopicExperimentResult, BERTopicHyperparameterCandidate, create_bertopic_experiment_storage_controller
from modules.topic.procedure.base import BERTopicIntermediateState, BERTopicProcedureComponent
from modules.topic.procedure.model_builder import BERTopicExperimentalModelBuilderProcedureComponent
from modules.topic.procedure.postprocess import BERTopicPostprocessProcedureComponent
from modules.topic.procedure.preprocess import BERTopicCacheOnlyPreprocessProcedureComponent, BERTopicDataLoaderProcedureComponent
from modules.topic.procedure.topic_modeling import BERTopicExperimentalTopicModelingProcedureComponent

logger = ProvisionedLogger().provision("Topic Modeling")


@dataclass
class BERTopicHyperparameterLab:
  task: TaskStorageProxy
  project_id: str
  column: str
  candidates: list[BERTopicHyperparameterCandidate]
  
  def run(self):
    config = ProjectCacheManager().get(self.project_id).config
    column = cast(TextualSchemaColumn, config.data_schema.assert_of_type(self.column, [SchemaColumnTypeEnum.Textual]))

    storage = create_bertopic_experiment_storage_controller(self.project_id, self.column)

    shared_state = BERTopicIntermediateState()
    shared_procedures = [
      BERTopicDataLoaderProcedureComponent(state=shared_state, task=self.task),
      BERTopicCacheOnlyPreprocessProcedureComponent(state=shared_state, task=self.task),
    ]

    for procedure in shared_procedures:
      procedure.run()

    for idx, candidate in enumerate(self.candidates):
      placeholder_task = TaskStorageProxy(
        id=f"Candidate {idx+1}",
        lock=threading.Lock(),
        results={}
      )
      # Shallow copy only, don't deep copy.
      state = copy(shared_state)
      state.column = candidate.apply(column)
      hash = BERTopicHyperparameterCandidate.hash(state.column.topic_modeling)
      # Already run before. We can skip.
      if storage.get(hash):
        self.task.log_success(f"Skipping running the topic model for Candidate {idx+1} since the hyperparameters had been evaluated before.")
        continue
      try:
        procedures: list[BERTopicProcedureComponent] = [
          BERTopicExperimentalModelBuilderProcedureComponent(state=state, task=placeholder_task, candidate=candidate),
          BERTopicExperimentalTopicModelingProcedureComponent(state=state, task=placeholder_task),
          BERTopicPostprocessProcedureComponent(state=state, task=placeholder_task, can_save=False),
        ]
        for procedure in procedures:
          procedure.run()
        evaluation = evaluate_topics(self.task, cast(Sequence[str], state.documents), state.result.topics, state.model)

        self.task.log_success(f"Finished running the topic model for Candidate {idx+1}")
        self.task.success(state.result)

        now = datetime.datetime.now()
        storage.update(hash, UserDataSchema(
          data=BERTopicExperimentResult(
            evaluation=evaluation,
            topic_modeling_config=state.column.topic_modeling,
            timestamp=now,
          ),
          name=f"Experiment {now.isoformat()}",
          description=None,
          tags=None,
        ), create_if_not_exist=True)
      except Exception as e:
        logger.exception(e)
        self.task.log_error(f"Failed to run an experiment on Candidate #{idx+1} due to the following error: {e}")
        pass

      return state
  
__all__ = [
  "BERTopicHyperparameterLab",
]