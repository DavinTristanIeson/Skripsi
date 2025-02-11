from dataclasses import dataclass
import functools
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, cast

import numpy as np

from controllers.topic.builder import BERTopicIndividualModels

if TYPE_CHECKING:
  from bertopic import BERTopic
  from scipy.sparse import spmatrix
  from sklearn.feature_extraction.text import CountVectorizer
  from bertopic.vectorizers import ClassTfidfTransformer

@dataclass
class BERTopicCTFIDFRepresentationResult:
  ctfidf: np.ndarray
  bow: np.ndarray
  words: list[tuple[str, float]]

@dataclass
class BERTopicTopicWordResult:
  topic_words: list[list[str]]
  labels: list[str]

@dataclass
class BERTopicInterpreter:
  vectorizer_model: CountVectorizer
  ctfidf_model: ClassTfidfTransformer
  # Always assume numpy array. Dealing with scipy sparse array typing is a pain.
  topic_ctfidf: np.ndarray
  top_n_words: int
  diversity: float

  documents: Optional[Sequence[str]]
  umap_embeddings: Optional[np.ndarray]

  @staticmethod
  def from_model(model: BERTopic, *, documents: Sequence[str])->"BERTopicInterpreter":
    bertopic_components = BERTopicIndividualModels.cast(model)
    return BERTopicInterpreter(
      ctfidf_model=bertopic_components.ctfidf_model,
      vectorizer_model=bertopic_components.vectorizer_model,
      topic_ctfidf=cast(np.ndarray, model.c_tf_idf_),
      top_n_words=model.top_n_words,
      diversity=bertopic_components.representation_model.diversity,
      umap_embeddings=bertopic_components.umap_model.load_cached_embeddings(),
      documents=documents,
    )
  
  def __mmr(self, target_ctfidf: np.ndarray, selected_words_ctfidf: np.ndarray):
    from bertopic.representation._mmr import mmr
    mmr_filtered_indices = mmr(
      doc_embedding=target_ctfidf,
      word_embeddings=selected_words_ctfidf,
      diversity=self.diversity,
      top_n=self.top_n_words,
      # Provide integers rather than strings so that MMR returns the indices
      words=np.arange(len(selected_words_ctfidf), dtype=np.int32) # type: ignore
    )
    return mmr_filtered_indices

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
    tuned_ctfidf = (global_ctfidf + ctfidf) / 2
    return tuned_ctfidf

  def get_weighted_words(self, ctfidf: np.ndarray)->list[tuple[str, float]]:
    top_word_indices = np.argsort(ctfidf)[:self.top_n_words] # type: ignore
    words = self.vocabulary[top_word_indices]
    weights = ctfidf[top_word_indices]

    mmr_filtered_indices = self.__mmr(ctfidf, weights)
    words = words[mmr_filtered_indices]
    weights = weights[mmr_filtered_indices]
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
    meta_document = ' '.join(documents)
    return cast(np.ndarray, self.vectorizer_model.transform(meta_document))
  
  def represent_as_ctfidf(self, bow: np.ndarray)->np.ndarray:
    return cast(np.ndarray, self.ctfidf_model.transform(bow)) # type: ignore
  
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


def bertopic_count_topics(model: "BERTopic")->int:
  return len(model.get_topics().keys()) - model._outliers