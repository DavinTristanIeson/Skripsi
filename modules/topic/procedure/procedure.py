from dataclasses import dataclass

from modules.logger import ProvisionedLogger
from modules.task import TaskManagerProxy
from modules.topic.procedure.model_builder import BERTopicModelBuilderProcedureComponent

from .base import BERTopicIntermediateState, BERTopicProcedureComponent
from .embedding import BERTopicEmbeddingProcedureComponent
from .postprocess import BERTopicPostprocessProcedureComponent, BERTopicVisualizationEmbeddingProcedureComponent
from .preprocess import BERTopicDataLoaderProcedureComponent, BERTopicPreprocessProcedureComponent
from .topic_modeling import BERTopicTopicModelingProcedureComponent

logger = ProvisionedLogger().provision("Topic Modeling")

@dataclass
class BERTopicProcedureFacade:
  task: TaskManagerProxy
  project_id: str
  column: str
  def run(self):
    state = BERTopicIntermediateState()
    
    procedures: list[BERTopicProcedureComponent] = [
      BERTopicDataLoaderProcedureComponent(state=state, task=self.task, project_id=self.project_id, column=self.column),
      BERTopicPreprocessProcedureComponent(state=state, task=self.task),
      BERTopicModelBuilderProcedureComponent(state=state, task=self.task),
      BERTopicEmbeddingProcedureComponent(state=state, task=self.task),
      BERTopicTopicModelingProcedureComponent(state=state, task=self.task),
      BERTopicVisualizationEmbeddingProcedureComponent(state=state, task=self.task),
      BERTopicPostprocessProcedureComponent(state=state, task=self.task),
    ]

    for procedure in procedures:
      procedure.run()
    
    # Column and config is loaded by DataLoaderProcedureComponent
    column = state.column
    config = state.config
    self.task.log_success(f"Finished discovering topics from \"{column.name}\" in Project \"{config.metadata.name}\" (data sourced from {config.source.path})")
    self.task.success(state.result)

    return state
  
__all__ = [
  "BERTopicProcedureFacade",
]