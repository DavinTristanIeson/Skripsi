from typing import TYPE_CHECKING, Sequence, cast

import numpy as np
import pandas as pd
from modules.topic.bertopic_ext.interpret import BERTopicInterpreter
from modules.topic.evaluation.method import cv_coherence, topic_diversity
from modules.topic.evaluation.model import TopicCoherenceV, TopicEvaluationResult, IndividualTopicEvaluationResult
from modules.topic.model import Topic

if TYPE_CHECKING:
  from bertopic import BERTopic

def evaluate_topics(
  documents: Sequence[str],
  topics: list[Topic],
  bertopic_model: "BERTopic",
  document_topic_assignments: pd.Series | np.ndarray,
  umap_vectors: np.ndarray
):
  topic_words: list[list[str]] = []
  for topic in topics:
    sorted_topic_words = sorted(topic.words, key=lambda word: word[1], reverse=True)
    top_n_topic_words = sorted_topic_words[:10]
    only_topic_words = list(map(lambda word: word[0], top_n_topic_words))
    topic_words.append(only_topic_words)
  interpreter = BERTopicInterpreter(bertopic_model)
  tokenized_documents = list(interpreter.tokenize(cast(Sequence[str], documents)))

  cv_score, cv_scores_per_topic_raw = cv_coherence(
    topic_words=topic_words,
    corpus=tokenized_documents
  )
  cv_scores_per_topic_nparray = np.array(cv_scores_per_topic_raw)

  topic_evaluation_results: list[IndividualTopicEvaluationResult] = []
  for topic_id, topic in enumerate(topics):
    coherence = cv_scores_per_topic_nparray[topic_id]
    topic_evaluation_results.append(IndividualTopicEvaluationResult(
      topic=topic,
      coherence=TopicCoherenceV(
        coherence=coherence[0],
        std_dev=coherence[1],
        support=coherence[2],
      )
    ))

  diversity = topic_diversity(
    topics=topic_words
  )
  
  outlier_count = (document_topic_assignments == -1).sum()
  total_count = len(document_topic_assignments)
  valid_count = total_count - outlier_count

  return TopicEvaluationResult(
    coherence_v=float(cv_score),
    topic_diversity=diversity,
    topics=topic_evaluation_results,
    
    outlier_count=outlier_count,
    valid_count=valid_count,
    total_count=total_count,
  )


__all__ = [
  "evaluate_topics"
]