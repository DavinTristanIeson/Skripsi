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
    return self.ctfidf_model
  
  @property
  def topic_ctfidf(self)->np.ndarray:
    # Always assume numpy array. Dealing with scipy sparse array typing is a pain.
    return cast(np.ndarray, self.model.c_tf_idf_)
  
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
  
  def get_words(self, ctfidf: np.ndarray)->list[str]:
    return list(map(lambda x: x[0], self.get_weighted_words(ctfidf)))

  def get_label(self, ctfidf: np.ndarray)->Optional[str]:
    representative_labels = list(filter(bool, self.get_words(ctfidf)))
    if len(representative_labels) == 0:
      return None
    return ', '.join(ctfidf)
  
  def tokenize(self, documents: Sequence[str])->Iterable[Sequence[str]]:
    analyzer = self.vectorizer_model.build_analyzer()
    return (analyzer(doc) for doc in documents)
  
  def represent_as_bow(self, documents: Sequence[str])->np.ndarray:
    return np.array(
      # This yields an np.matrix, which is why we need to get [0]
      self.vectorizer_model.transform(documents).sum(axis=0) # type: ignore
    )[0]
  
  def represent_as_ctfidf(self, bow: np.ndarray)->np.ndarray:
    return cast(np.ndarray, self.ctfidf_model.transform([bow]))[0] # type: ignore
  
  def extract_topics(self)->list[Topic]:
    model = self.model
    if model.topic_embeddings_ is None:
      raise ApiError(
        "BERTopic model has not been fitted yet! This might be a developer oversight.",
        http.HTTPStatus.INTERNAL_SERVER_ERROR
      )
    topic_words_mapping = model.get_topics()
    topics: list[Topic] = []
    for raw_key, raw_topic_words in topic_words_mapping.items():
      key = int(raw_key)
      if key == -1:
        continue
      topic_words = cast(list[tuple[str, float]], raw_topic_words)
      topic_words = list(filter(lambda x: len(x[0]) > 0, topic_words))

      representative_topic_words = list(itertools.islice(map(
        lambda el: el[0],
        topic_words
      ), 3))
      if len(representative_topic_words) == 0:
        topic_label = f"Topic {key+1}"
      else:
        topic_label = ', '.join(representative_topic_words)
      topic_frequency = cast(int, model.get_topic_freq(int(key)))

      topic = Topic(
        id=key,
        label=topic_label,
        words=topic_words,
        frequency=topic_frequency,
      )
      topics.append(topic)

    return topics

  @property
  def topic_count(self)->int:
    return len(self.model.get_topics().keys()) - self.model._outliers
  
  @property
  def topic_embeddings(self)->np.ndarray:
    return self.model.topic_embeddings_[self.model._outliers:] # type: ignore
  
  def topic_ctfidfs_per_class(self, ctfidf: np.ndarray, document_topic_assignments: np.ndarray | pd.Series):
    from sklearn.preprocessing import normalize

    # Normalize. This follows the code in BERTopic implementation
    global_ctfidf = normalize(self.topic_ctfidf, axis=1, norm="l1", copy=False)
    local_ctfidf = normalize(ctfidf, axis=1, norm="l1", copy=False)

    # Filter out invalid topics
    valid_topic_mask = np.bitwise_and(document_topic_assignments >= 0, document_topic_assignments < self.topic_ctfidf.shape[0])
    document_topic_assignments = document_topic_assignments[valid_topic_mask]

    topic_assignment_aggregation = pd.Series(document_topic_assignments).value_counts(normalize=True)
    unique_topics = topic_assignment_aggregation.index
    topic_proportions = topic_assignment_aggregation.values

    # Get the average between global and local. This follows the code in BERTopic implementation
    # Except for the topic proportions part. That one is ours.
    global_ctfidf = global_ctfidf.take(unique_topics, axis=0) * topic_proportions
    tuned_ctfidf = (global_ctfidf + local_ctfidf ) / 2
    return tuned_ctfidf, unique_topics

__all__ = [
  "BERTopicInterpreter",
]