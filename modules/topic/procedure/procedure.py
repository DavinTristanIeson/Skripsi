from dataclasses import dataclass
from typing import cast

from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.logger import ProvisionedLogger
from modules.project.cache import ProjectCacheManager
from modules.task import TaskStorageProxy
from modules.topic.bertopic_ext.builder import BERTopicModelBuilder
from modules.topic.procedure.model_builder import BERTopicModelBuilderProcedureComponent

from .base import BERTopicIntermediateState, BERTopicProcedureComponent
from .embedding import BERTopicEmbeddingProcedureComponent
from .postprocess import BERTopicPostprocessProcedureComponent, BERTopicVisualizationEmbeddingProcedureComponent
from .preprocess import BERTopicDataLoaderProcedureComponent, BERTopicPreprocessProcedureComponent
from .topic_modeling import BERTopicTopicModelingProcedureComponent

logger = ProvisionedLogger().provision("Topic Modeling")

@dataclass
class BERTopicProcedureFacade:
  task: TaskStorageProxy
  project_id: str
  column: str
  def run(self):
    config = ProjectCacheManager().get(self.project_id).config
    column = config.data_schema.assert_of_type(self.column, [SchemaColumnTypeEnum.Textual])

    state = BERTopicIntermediateState()
    state.config = config
    state.column = cast(TextualSchemaColumn, column)
    procedures: list[BERTopicProcedureComponent] = [
      BERTopicDataLoaderProcedureComponent(state=state, task=self.task),
      BERTopicPreprocessProcedureComponent(state=state, task=self.task),
      BERTopicModelBuilderProcedureComponent(state=state, task=self.task),
      BERTopicEmbeddingProcedureComponent(state=state, task=self.task),
      BERTopicTopicModelingProcedureComponent(state=state, task=self.task),
      BERTopicVisualizationEmbeddingProcedureComponent(state=state, task=self.task),
      BERTopicPostprocessProcedureComponent(state=state, task=self.task),
    ]

    for procedure in procedures:
      procedure.run()
    self.task.log_success(f"Finished discovering topics from \"{column.name}\" in Project \"{config.metadata.name}\" (data sourced from {config.source.path})")
    self.task.success(state.result)

    return state
  
__all__ = [
  "BERTopicProcedureFacade",
]