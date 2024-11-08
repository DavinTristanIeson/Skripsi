import itertools
from typing import TYPE_CHECKING, Optional, Sequence, cast
if TYPE_CHECKING:
    from bertopic import BERTopic

def bertopic_topic_words(model: "BERTopic")->list[list[str]]:
  bertopics = model.get_topics()
  topic_words: list[list[str]] = []
  for key, distribution in bertopics.items():
    if key == -1:
      continue
    topic_distribution = cast(list[tuple[str, float]], distribution)
    topic_words.append(list(map(
      lambda el: el[0],
      topic_distribution
    )))

  return topic_words

def bertopic_topic_labels(model: "BERTopic")->list[str]:
  topic_words = bertopic_topic_words(model)
  topic_labels = list(map(lambda words: ', '.join(filter(bool, words[:5])), topic_words))
  custom_labels_maybe = cast(Optional[list[str]], model.custom_labels_)

  if custom_labels_maybe is not None:
    custom_labels = custom_labels_maybe[model._outliers:]
  else:
    custom_labels = list(itertools.repeat('', len(topic_labels)))
    
  labels: list[str] = list(map(
    lambda idx, generated, custom: custom or generated or f"Unnamed Topic {idx + 1}",
    range(len(topic_labels)), topic_labels, custom_labels
  ))

  return labels

def bertopic_vectorize(model: "BERTopic", documents: Sequence[str])->tuple[Sequence[Sequence[str]], Sequence[str]]:
  vectorizer = model.vectorizer_model
  analyzer = vectorizer.build_analyzer()
  words = cast(Sequence[str], vectorizer.get_feature_names_out())
  tokens = [analyzer(doc) for doc in documents]
  return tokens, words