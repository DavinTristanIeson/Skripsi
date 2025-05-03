import datetime

import pydantic

from modules.topic.model import Topic


class CoherenceVPerTopic(pydantic.BaseModel):
  topic: Topic
  coherence: float
  std_dev: float
  support: int
  

class BERTopicEvaluationResult(pydantic.BaseModel):
  coherence_v: float
  topic_diversity: float
  coherence_v_per_topic: list[CoherenceVPerTopic]

class BERTopicEvaluationPayload(pydantic.BaseModel):
  column: str
  top_n_words: int
  
__all__ = [
  "BERTopicEvaluationResult"
]