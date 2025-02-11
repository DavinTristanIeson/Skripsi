from dataclasses import dataclass
import functools
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, cast

import numpy as np
import numpy.typing as npt

from models.topic.topic import TopicModel
if TYPE_CHECKING:
  from bertopic import BERTopic
  from scipy.sparse import spmatrix
  from sklearn.feature_extraction.text import CountVectorizer
  from bertopic.vectorizers import ClassTfidfTransformer

@dataclass
class BERTopicCTFIDFRepresentationResult:
  ctfidf: npt.NDArray
  bow: npt.NDArray
  words: list[tuple[str, float]]

@dataclass
class BERTopicTopicWordResult:
  topic_words: list[list[str]]
  labels: list[str]

@dataclass
class BERTopicInterpreter:
  vectorizer_model: CountVectorizer
  ctfidf_model: ClassTfidfTransformer
  topic_ctfidf: npt.NDArray
  n_words: int

  @functools.cached_property
  def vocabulary(self):
    return self.vectorizer_model.get_feature_names_out()

  def get_weighted_words(self, ctfidf: npt.NDArray)->list[tuple[str, float]]:
    top_word_indices = np.argsort(ctfidf)[:self.n_words] # type: ignore
    words = self.vocabulary[top_word_indices]
    weights = ctfidf[top_word_indices]
    return list(zip(words, weights))
  
  def get_words(self, ctfidf: npt.NDArray)->list[str]:
    return list(map(lambda x: x[0], self.get_weighted_words(ctfidf)))

  def get_label(self, ctfidf: npt.NDArray)->Optional[str]:
    representative_labels = list(filter(bool, self.get_words(ctfidf)))
    if len(representative_labels) == 0:
      return None
    return ', '.join(ctfidf)
  
  def tokenize(self, documents: Sequence[str])->Iterable[str]:
    analyzer = self.vectorizer_model.build_analyzer()
    return (analyzer(doc) for doc in documents)
  
  def represent_as_bow(self, documents: Sequence[str])->npt.NDArray:
    meta_document = ' '.join(documents)
    return cast(npt.NDArray, self.vectorizer_model.transform(meta_document))
  
  def represent_as_ctfidf(self, bow: npt.NDArray)->npt.NDArray:
    return cast(npt.NDArray, self.ctfidf_model.transform(bow)) # type: ignore
  
def bertopic_topic_words(model: BERTopic):
  topic_words_mapping = model.get_topics()
  all_topic_words: list[list[str]] = []
  for key, raw_topic_words in topic_words_mapping.items():
    if key == -1:
      continue
    raw_topic_words = cast(list[tuple[str, float]], raw_topic_words)
    all_topic_words.append(list(map(
      lambda el: el[0],
      raw_topic_words
    )))

  topic_labels = []
  for idx, topic_words in enumerate(all_topic_words):
    representative_topic_words = list(filter(bool, topic_words[:3]))
    if len(representative_topic_words) == 0:
      topic_labels.append(f"Topic {idx+1}")
    else:
      topic_labels.append(', '.join(representative_topic_words))

  return BERTopicTopicWordResult(
    topic_words=all_topic_words,
    labels=topic_labels
  )
