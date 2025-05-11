from typing import TYPE_CHECKING, Sequence, cast

import numpy as np
import pandas as pd
from modules.topic.bertopic_ext.interpret import BERTopicInterpreter
from modules.topic.evaluation.method import cv_coherence, topic_diversity
from modules.topic.evaluation.model import TopicEvaluationResult, CoherenceVPerTopic
from modules.topic.model import Topic

if TYPE_CHECKING:
  from bertopic import BERTopic

def evaluate_topics(
  raw_documents: Sequence[str],
  topics: list[Topic],
  bertopic_model: "BERTopic",
  document_topic_assignments: pd.Series | np.ndarray
):
  topic_words: list[list[str]] = []
  for topic in topics:
    sorted_topic_words = sorted(topic.words, key=lambda word: word[1], reverse=True)
    top_n_topic_words = sorted_topic_words[:10]
    only_topic_words = list(map(lambda word: word[0], top_n_topic_words))
    topic_words.append(only_topic_words)
  interpreter = BERTopicInterpreter(bertopic_model)
  documents = list(interpreter.tokenize(cast(Sequence[str], raw_documents)))

  cv_score, cv_scores_per_topic_raw = cv_coherence(topic_words, documents)
  cv_scores_per_topic_nparray = np.array(cv_scores_per_topic_raw)

  cv_scores_per_topic = list(map(
    lambda topic, coherence, std, support: CoherenceVPerTopic(topic=topic, coherence=coherence, std_dev=std, support=support),
    topics, cv_scores_per_topic_nparray[:, 0], cv_scores_per_topic_nparray[:, 1], cv_scores_per_topic_nparray[:, 2]
  ))

  diversity = topic_diversity(topic_words)

  mask = pd.notna(document_topic_assignments)
  document_topic_assignments = document_topic_assignments[mask]
  outlier_count = (document_topic_assignments == -1).sum()
  total_count = len(document_topic_assignments)
  valid_count = total_count - outlier_count

  return TopicEvaluationResult(
    coherence_v=cv_score,
    topic_diversity=diversity,
    coherence_v_per_topic=cv_scores_per_topic,
    outlier_count=outlier_count,
    valid_count=valid_count,
    total_count=total_count,
  )


__all__ = [
  "evaluate_topics"
]