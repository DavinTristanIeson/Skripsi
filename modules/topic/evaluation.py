from typing import Sequence

import pydantic

class ColumnTopicsEvaluationResult(pydantic.BaseModel):
  column: str
  topics: Sequence[str]
  cv_score: float
  topic_diversity_score: float
  cv_topic_scores: Sequence[float]
  cv_barchart: str = pydantic.Field(repr=False)

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
  td = len(unique_words) / total_words
  return td

def cv_coherence(topic_words: Sequence[Sequence[str]], corpus: Sequence[Sequence[str]])->tuple[float, Sequence[float]]:
  from gensim.models.coherencemodel import CoherenceModel
  from gensim.corpora import Dictionary

  dictionary = Dictionary()
  dictionary.add_documents(corpus)
  cv_coherence = CoherenceModel(
    topics=topic_words,
    texts=corpus,
    coherence='c_v',
    dictionary=dictionary,
  )
  cv_score = cv_coherence.get_coherence()
  cv_scores_per_topic = cv_coherence.get_coherence_per_topic(
    with_std=True,
    with_support=True,
  )

  return cv_score, cv_scores_per_topic

__all__ = [
  "ColumnTopicsEvaluationResult",
  "ProjectTopicsEvaluationResult",
  "topic_diversity",
  "cv_coherence"
]