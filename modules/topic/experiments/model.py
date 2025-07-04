import datetime
from typing import TYPE_CHECKING, Annotated, Optional

import pydantic

from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.topic.evaluation.model import TopicEvaluationResult
from modules.topic.exceptions import MissingHyperparameterOptimizationConstraintsException
from modules.validation.array import StartBeforeEndValidator

if TYPE_CHECKING:
  from optuna.trial import Trial


class BERTopicHyperparameterCandidate(pydantic.BaseModel):
  min_topic_size: Optional[int]
  max_topics: Optional[int]
  topic_confidence_threshold: Optional[int]

  def apply(self, column: TextualSchemaColumn, copy: bool):
    if copy:
      column = column.model_copy(deep=True)
    if self.min_topic_size is not None:
      column.topic_modeling.min_topic_size = self.min_topic_size
    if self.max_topics is not None:
      column.topic_modeling.max_topics = self.max_topics
    if self.topic_confidence_threshold is not None:
      column.topic_modeling.topic_confidence_threshold = self.topic_confidence_threshold
    return column

class BERTopicHyperparameterConstraint(pydantic.BaseModel):
  min_topic_size: Annotated[Optional[tuple[int, int]], StartBeforeEndValidator]
  topic_confidence_threshold: Annotated[Optional[tuple[int, int]], StartBeforeEndValidator]
  topic_count: Annotated[Optional[tuple[int, int]], StartBeforeEndValidator]

  @pydantic.model_validator(mode="after")
  def __validate_constraints(self):
    if (
      self.min_topic_size is None and\
      self.topic_confidence_threshold is None
    ):
      raise MissingHyperparameterOptimizationConstraintsException("Provide constraints for at least min. topic size or topic confidence threshold for the experiments.")
    return self

  def suggest(self, trial: "Trial")->BERTopicHyperparameterCandidate:
    min_topic_size = self.min_topic_size and trial.suggest_int(
      "min_topic_size", self.min_topic_size[0], self.min_topic_size[1]
    )
    topic_confidence_threshold = self.topic_confidence_threshold and trial.suggest_int(
      "topic_confidence_threshold", self.topic_confidence_threshold[0], self.topic_confidence_threshold[1]
    )
    return BERTopicHyperparameterCandidate(
      topic_confidence_threshold=topic_confidence_threshold,
      max_topics=self.topic_count and self.topic_count[1],
      min_topic_size=min_topic_size
    )

class BERTopicExperimentTrialResult(pydantic.BaseModel):
  trial_number: int
  candidate: BERTopicHyperparameterCandidate
  evaluation: Optional[TopicEvaluationResult]
  optuna_metric: float
  error: Optional[str]
  timestamp: datetime.datetime = pydantic.Field(default_factory=lambda: datetime.datetime.now())

class BERTopicExperimentResult(pydantic.BaseModel):
  constraint: BERTopicHyperparameterConstraint
  max_trials: int
  trials: list[BERTopicExperimentTrialResult]
  evaluation: TopicEvaluationResult
  start_at: datetime.datetime
  end_at: Optional[datetime.datetime]
  last_updated_at: datetime.datetime

__all__ = [
  "BERTopicHyperparameterConstraint",
  "BERTopicExperimentTrialResult",
]