from dataclasses import dataclass
import datetime

import pydantic

from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.config.schema.textual import TopicModelingConfig
from modules.project.paths import ProjectPathManager, ProjectPaths
from modules.storage.userdata.filesystem import UserDataStorageController
from modules.storage.userdata.resource import UserDataResource
from modules.topic.evaluation.model import BERTopicEvaluationResult

class BERTopicHyperparameterCandidate(pydantic.BaseModel):
  min_topic_size: int
  max_topics: int
  clustering_conservativeness: float

  def apply(self, column: TextualSchemaColumn):
    column = column.model_copy()
    column.topic_modeling.clustering_conservativeness = self.clustering_conservativeness
    column.topic_modeling.max_topics = self.max_topics
    column.topic_modeling.min_topic_size = self.min_topic_size
    return column

  @staticmethod
  def hash(topic_modeling_config: TopicModelingConfig):
    import hashlib
    return hashlib.md5(str(topic_modeling_config).encode('utf-8')).hexdigest()
  
class BERTopicExperimentResult(pydantic.BaseModel):
  topic_modeling_config: TopicModelingConfig
  evaluation: BERTopicEvaluationResult
  timestamp: datetime.datetime = pydantic.Field(default_factory=lambda: datetime.datetime.now())

def create_bertopic_experiment_storage_controller(project_id: str, column: str):
  paths = ProjectPathManager(project_id=project_id)
  experiment_result_path = paths.full_path(ProjectPaths.BERTopicExperiments(column))
  storage = UserDataStorageController[BERTopicExperimentResult](
    path=experiment_result_path,
    validator=lambda data: UserDataResource[BERTopicExperimentResult].model_validate(data)
  )
  return storage

__all__ = [
  "BERTopicHyperparameterCandidate",
  "BERTopicExperimentResult",
  "create_bertopic_experiment_storage_controller"
]