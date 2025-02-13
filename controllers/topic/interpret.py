from dataclasses import dataclass
import functools
import itertools
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, cast
import http

import numpy as np

from common.models.api import ApiError
from controllers.topic.builder import BERTopicIndividualModels
from models.topic.topic import TopicModel

if TYPE_CHECKING:
  from bertopic import BERTopic
  from sklearn.feature_extraction.text import CountVectorizer
  from bertopic.vectorizers import ClassTfidfTransformer

@dataclass
class BERTopicCTFIDFRepresentationResult:
  ctfidf: np.ndarray
  bow: np.ndarray
  words: list[tuple[str, float]]

@dataclass
class BERTopicInterpreter:
  vectorizer_model: "CountVectorizer"
  ctfidf_model: "ClassTfidfTransformer"
  # Always assume numpy array. Dealing with scipy sparse array typing is a pain.
  topic_ctfidf: np.ndarray
  top_n_words: int

  @staticmethod
  def from_model(model: "BERTopic")->"BERTopicInterpreter":
    bertopic_components = BERTopicIndividualModels.cast(model)
    return BERTopicInterpreter(
      ctfidf_model=bertopic_components.ctfidf_model,
      vectorizer_model=bertopic_components.vectorizer_model,
      topic_ctfidf=cast(np.ndarray, model.c_tf_idf_),
      top_n_words=model.top_n_words,
    )

  @functools.cached_property
  def vocabulary(self):
    return self.vectorizer_model.get_feature_names_out()
  
  def tuning_topic_ctfidf(self, ctfidf: np.ndarray, involved_topics: np.ndarray):
    from sklearn.preprocessing import normalize

    # Normalize. This follows the code in BERTopic implementation
    global_ctfidf = normalize(self.topic_ctfidf, axis=1, norm="l1", copy=False)
    local_ctfidf = normalize(ctfidf, axis=1, norm="l1", copy=False)

    # Filter out invalid topics
    valid_topic_mask = np.bitwise_and(involved_topics >= 0, involved_topics < self.topic_ctfidf.shape[0])
    involved_topics = involved_topics[valid_topic_mask]

    # Get the average between global and local. This follows the code in BERTopic implementation
    global_ctfidf = global_ctfidf.take(involved_topics, axis=0)
    tuned_ctfidf = (global_ctfidf + local_ctfidf ) / 2
    return tuned_ctfidf

  def get_weighted_words(self, ctfidf: np.ndarray)->list[tuple[str, float]]:
    top_word_indices = np.argsort(ctfidf)[:self.top_n_words] # type: ignore
    words = self.vocabulary[top_word_indices]
    weights = ctfidf[top_word_indices]
    return list(zip(words, weights))
  
  def get_words(self, ctfidf: np.ndarray)->list[str]:
    return list(map(lambda x: x[0], self.get_weighted_words(ctfidf)))

  def get_label(self, ctfidf: np.ndarray)->Optional[str]:
    representative_labels = list(filter(bool, self.get_words(ctfidf)))
    if len(representative_labels) == 0:
      return None
    return ', '.join(ctfidf)
  
  def tokenize(self, documents: Sequence[str])->Iterable[str]:
    analyzer = self.vectorizer_model.build_analyzer()
    return (analyzer(doc) for doc in documents)
  
  def represent_as_bow(self, documents: Sequence[str])->np.ndarray:
    return np.array(self.vectorizer_model.transform(documents).sum(axis=0))[0] # type: ignore
  
  def represent_as_ctfidf(self, bow: np.ndarray)->np.ndarray:
    return cast(np.ndarray, self.ctfidf_model.transform([bow]))[0] # type: ignore
  

def bertopic_extract_topics(
  model: "BERTopic",
  visualization_topic_embeddings: np.ndarray
)->list[TopicModel]:
  if model.topic_embeddings_ is None:
    raise ApiError(
      "BERTopic model has not been fitted yet! This might be a developer oversight.",
      http.HTTPStatus.INTERNAL_SERVER_ERROR
    )
  topic_words_mapping = model.get_topics()
  topics: list[TopicModel] = []
  for raw_key, raw_topic_words in topic_words_mapping.items():
    key = int(raw_key)
    if key == -1:
      continue
    topic_words = cast(list[tuple[str, float]], raw_topic_words)
    topic_words = list(filter(lambda x: len(x[0]) > 0, topic_words))
    topic_embedding = model.topic_embeddings_[key] # type: ignore
    pythonic_topic_embedding = list(map(float, topic_embedding))
    visualization_topic_embedding = visualization_topic_embeddings[key]
    pythonic_visualization_topic_embedding = list(map(float, visualization_topic_embedding))

    representative_topic_words = list(itertools.islice(map(
      lambda el: el[0],
      topic_words
    ), 3))
    if len(representative_topic_words) == 0:
      topic_label = f"Topic {key+1}"
    else:
      topic_label = ', '.join(representative_topic_words)
    topic_frequency = cast(int, model.get_topic_freq(int(key)))

    topic = TopicModel(
      id=key,
      label=topic_label,
      words=topic_words,
      frequency=topic_frequency,
      visualization_embedding=pythonic_visualization_topic_embedding,
    )
    topics.append(topic)

  return topics

def bertopic_count_topics(model: "BERTopic")->int:
  return len(model.get_topics().keys()) - model._outliers