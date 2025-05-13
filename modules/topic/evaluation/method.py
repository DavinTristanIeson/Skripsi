
from typing import Sequence, cast

import numpy as np

from modules.topic.model import Topic


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
  from gensim.corpora import Dictionary
  from gensim.models.coherencemodel import CoherenceModel
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

def silhouette_score(topics: list[Topic], document_topic_assignments: np.ndarray, umap_vectors: np.ndarray):
  import scipy.spatial.distance

  silhouette_scores = []
  excluded_mask = document_topic_assignments == -1
  for topic in topics:
    cluster_mask = document_topic_assignments == topic.id
    cluster_data = umap_vectors[cluster_mask]
    non_cluster_data = umap_vectors[(~cluster_mask) & (~excluded_mask)]
    if len(cluster_data) == 0 or len(non_cluster_data) == 0:
      silhouette_scores.append(0)
      continue

    intra_cluster = scipy.spatial.distance.cdist(cluster_data, cluster_data, metric="euclidean")
    inter_cluster = scipy.spatial.distance.cdist(cluster_data, non_cluster_data, metric="euclidean")

    a_c = np.mean(intra_cluster[intra_cluster != 0])
    b_c = np.min(inter_cluster)
    s_c = (b_c - a_c) / np.max([a_c, b_c])

    silhouette_scores.append(s_c)
    
  return np.mean(silhouette_scores), silhouette_scores

__all__ = [
  "cv_coherence",
  "topic_diversity",
  "silhouette_score"
]