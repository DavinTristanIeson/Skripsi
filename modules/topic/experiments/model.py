import datetime
from typing import TYPE_CHECKING, Annotated, Optional

import pydantic

from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.config.schema.textual import TopicModelingConfig
from modules.project.paths import ProjectPathManager, ProjectPaths
from modules.storage.userdata.filesystem import UserDataStorageController
from modules.storage.userdata.resource import UserDataResource
from modules.topic.evaluation.model import TopicEvaluationResult
from modules.validation.array import StartBeforeEndValidator

if TYPE_CHECKING:
  from optuna.trial import Trial


class BERTopicHyperparameterCandidate(pydantic.BaseModel):
  min_topic_size: Optional[int]
  max_topics: Optional[int]
  clustering_conservativeness: Optional[float]

  def apply(self, column: TextualSchemaColumn):
    column = column.model_copy(deep=True)
    if self.min_topic_size is not None:
      column.topic_modeling.min_topic_size = self.min_topic_size
    if self.max_topics is not None:
      column.topic_modeling.max_topics = self.max_topics
    if self.clustering_conservativeness is not None:
      column.topic_modeling.clustering_conservativeness = self.clustering_conservativeness
    return column

class BERTopicHyperparameterConstraint(pydantic.BaseModel):
  min_topic_size: Annotated[Optional[tuple[int, int]], StartBeforeEndValidator]
  max_topics: Annotated[Optional[tuple[int, int]], StartBeforeEndValidator]
  clustering_conservativeness: Annotated[Optional[tuple[float, float]], StartBeforeEndValidator]

  def suggest(self, trial: "Trial")->BERTopicHyperparameterCandidate:
    min_topic_size = self.min_topic_size and trial.suggest_int(
      "min_topic_size", self.min_topic_size[0], self.min_topic_size[1]
    )
    max_topics = self.max_topics and trial.suggest_int(
      "max_topics", self.max_topics[0], self.max_topics[1]
    )
    clustering_conservativeness = self.clustering_conservativeness and trial.suggest_float(
      "clustering_conservativeness", self.clustering_conservativeness[0], self.clustering_conservativeness[1]
    )
    return BERTopicHyperparameterCandidate(
      clustering_conservativeness=clustering_conservativeness,
      max_topics=max_topics,
      min_topic_size=min_topic_size
    )

class BERTopicExperimentTrialResult(pydantic.BaseModel):
  topic_modeling_config: TopicModelingConfig
  evaluation: Optional[TopicEvaluationResult]
  error: Optional[str]
  timestamp: datetime.datetime = pydantic.Field(default_factory=lambda: datetime.datetime.now())

class BERTopicExperimentResult(pydantic.BaseModel):
  trials: list[BERTopicExperimentTrialResult]
  start_at: datetime.datetime
  end_at: Optional[datetime.datetime]
  last_updated_at: datetime.datetime

__all__ = [
  "BERTopicHyperparameterConstraint",
  "BERTopicExperimentTrialResult",
]