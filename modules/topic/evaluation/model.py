import pydantic

from modules.topic.model import Topic

class CoherenceVPerTopic(pydantic.BaseModel):
  topic: Topic
  coherence: float
  std_dev: float
  support: int
  
class TopicEvaluationResult(pydantic.BaseModel):
  coherence_v: float
  topic_diversity: float
  coherence_v_per_topic: list[CoherenceVPerTopic]
  outlier_count: int

__all__ = [
  "TopicEvaluationResult"
]