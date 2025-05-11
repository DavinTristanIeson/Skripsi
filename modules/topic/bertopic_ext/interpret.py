from dataclasses import dataclass
import functools
import itertools
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, cast
import http

import numpy as np
import pandas as pd

from modules.api import ApiError

from .builder import BERTopicIndividualModels
from ..model import Topic

import scipy.sparse
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
  from bertopic import BERTopic
  from sklearn.feature_extraction.text import CountVectorizer
  from bertopic.vectorizers import ClassTfidfTransformer

@dataclass
class BERTopicInterpreter:
  model: "BERTopic"

  @property
  def vectorizer_model(self)->"CountVectorizer":
    return self.model.vectorizer_model
  
  @property
  def ctfidf_model(self)->"ClassTfidfTransformer":
    return self.model.ctfidf_model # type: ignore
  
  @property
  def topic_ctfidf(self)->csr_matrix:
    return self.model.c_tf_idf_[self.model._outliers:] # type: ignore
  
  @functools.cached_property
  def top_n_words(self):
    return self.model.top_n_words

  @functools.cached_property
  def vocabulary(self):
    return self.vectorizer_model.get_feature_names_out()

  def get_weighted_words(self, ctfidf: np.ndarray)->list[tuple[str, float]]:
    # Very odd logic but trust me this works. This gets the last n elements from argsort.
    top_word_indices = np.argsort(ctfidf)[:self.top_n_words+1:-1]

    # Don't get words where the C-TF-IDF is 0.
    valid_top_word_indices = top_word_indices[ctfidf[top_word_indices] > 0]
    if len(valid_top_word_indices) == 0:
      return []

    words = self.vocabulary[valid_top_word_indices]
    weights = ctfidf[valid_top_word_indices]
    return list(zip(words, weights))

  def unweigh_words(self, weighted_words: list[tuple[str, float]])->list[str]:
    return list(map(lambda x: x[0], weighted_words))

  def get_label(self, weighted_words: list[tuple[str,float]])->Optional[str]:
    representative_labels = list(filter(bool, self.unweigh_words(weighted_words)))[:3]
    if len(representative_labels) == 0:
      return None
    return ', '.join(representative_labels)
  
  def tokenize(self, documents: Sequence[str])->Iterable[Sequence[str]]:
    analyzer = self.vectorizer_model.build_analyzer()
    return (analyzer(doc) for doc in documents)
  
  def represent_as_bow(self, documents: Sequence[str])->np.ndarray:
    # BERTopic's C-TF-IDF implementation uses csr_matrix rather than csr_array.
    return np.array(self.vectorizer_model.transform(documents).sum(axis=0))[0] # type: ignore
  
  def represent_as_bow_sparse(self, documents: Sequence[str])->csr_matrix:
    # BERTopic's C-TF-IDF implementation uses csr_matrix rather than csr_array.
    return cast(np.ndarray, csr_matrix(self.vectorizer_model.transform(documents).sum(axis=0))) # type: ignore
  
  def represent_as_ctfidf(self, bow: csr_matrix)->csr_matrix:
    return self.ctfidf_model.transform(bow)
  
  def extract_topics(self, *, map_topics: bool = False)->list[Topic]:
    model = self.model
    topic_words_mapping = model.get_topics()
    if map_topics and model.topic_mapper_ is not None:
      topic_mapping = model.topic_mapper_.get_mappings()
    else:
      topic_mapping = None
    topics: list[Topic] = []
    for raw_key, raw_topic_words in topic_words_mapping.items():
      key = int(raw_key)
      # Don't store outliers as a topic
      if key == -1:
        continue
      if map_topics and topic_mapping is not None:
        if key not in topic_mapping:
          continue
        # get original Y value
        key = topic_mapping[key]

      topic_words = cast(list[tuple[str, float]], raw_topic_words)
      topic_words = list(filter(lambda x: len(x[0]) > 0, topic_words))
      topic_frequency = cast(int, model.get_topic_freq(int(key)))

      topic = Topic(
        id=key,
        label=None,
        words=topic_words,
        frequency=topic_frequency,
      )

      # Don't store non-existent topics
      if topic_frequency == 0:
        continue
      topics.append(topic)

    return sorted(topics, key=lambda topic: topic.id)

  @property
  def topic_count(self)->int:
    return len(self.model.get_topics().keys()) - self.model._outliers
  
  @property
  def topic_embeddings(self)->np.ndarray:
    return self.model.topic_embeddings_[self.model._outliers:] # type: ignore
  
  def topic_ctfidfs_per_class(self, ctfidf: csr_matrix, document_topic_assignments: np.ndarray | pd.Series)->tuple[csr_matrix, Sequence[int]]:
    from sklearn.preprocessing import normalize

    # Normalize. This follows the code in BERTopic implementation
    global_ctfidf = normalize(self.topic_ctfidf, axis=1, norm="l1", copy=False)
    if len(document_topic_assignments) == 0:
      return global_ctfidf, []
    # Turn the shape into (1, ctfidf.shape[0])
    local_ctfidf = normalize(ctfidf, axis=1, norm="l1", copy=False)

    # Filter out invalid topics
    topic_counts = global_ctfidf.shape[0]
    valid_topic_mask = np.bitwise_and(document_topic_assignments >= 0, document_topic_assignments < topic_counts)
    document_topic_assignments = document_topic_assignments[valid_topic_mask]

    topic_assignment_aggregation = pd.Series(document_topic_assignments).value_counts(normalize=True)
    unique_topics = topic_assignment_aggregation.index

    tuned_ctfidf_raw = []
    for row in global_ctfidf:
      tuned_ctfidf = (row + local_ctfidf) / 2
      tuned_ctfidf_raw.append(tuned_ctfidf)
    tuned_ctfidf = csr_matrix(scipy.sparse.vstack(tuned_ctfidf_raw))
    return tuned_ctfidf, cast(Sequence[int], unique_topics)

__all__ = [
  "BERTopicInterpreter",
]