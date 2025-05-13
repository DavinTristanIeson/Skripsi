import pydantic

from modules.topic.model import Topic

class TopicCoherenceV(pydantic.BaseModel):
  coherence: float
  std_dev: float
  support: int

class IndividualTopicEvaluationResult(pydantic.BaseModel):
  topic: Topic
  coherence: TopicCoherenceV
  silhouette_score: float
  
class TopicEvaluationResult(pydantic.BaseModel):
  coherence_v: float
  silhouette_score: float
  topic_diversity: float
  silhouette_score: float
  topics: list[IndividualTopicEvaluationResult]
  outlier_count: int
  valid_count: int
  total_count: int

__all__ = [
  "TopicEvaluationResult"
]