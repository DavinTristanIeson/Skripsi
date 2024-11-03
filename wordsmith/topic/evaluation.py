from typing import Mapping, Sequence

import pydantic
from gensim.models.coherencemodel import CoherenceModel

class ColumnTopicsEvaluationResult(pydantic.BaseModel):
  column: str
  topics: Sequence[str]
  cv_score: float
  topic_diversity_score: float
  cv_topic_scores: Sequence[float]

class ProjectTopicsEvaluationResult(pydantic.RootModel):
  root: dict[str, ColumnTopicsEvaluationResult]

def topic_diversity(topics: Sequence[Sequence[str]]):
  # based on OCTIS implementation and the equation in https://www.researchgate.net/publication/343173999_Topic_Modeling_in_Embedding_Spaces
  # https://github.com/MIND-Lab/OCTIS/blob/master/octis/evaluation_metrics/diversity_metrics.py#L12
  total_words = 0
  unique_words: set[str] = set()
  for topic in topics:
    unique_words |= set(topic)
    total_words += len(topic)
  td = (1 - (len(unique_words) / total_words)) ** 2
  return td

def cv_coherence(topic_words: Sequence[Sequence[str]], corpus: Sequence[Sequence[str]])->tuple[float, Sequence[float]]:
  cv_coherence = CoherenceModel(
    topics=topic_words,
    corpus=corpus,
    coherence='c_v'
  )
  cv_score = cv_coherence.get_coherence()
  cv_scores_per_topic = cv_coherence.get_coherence_per_topic(
    with_std=True,
    with_support=True,
  )

  return cv_score, cv_scores_per_topic