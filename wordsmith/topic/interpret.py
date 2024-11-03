from typing import TYPE_CHECKING, Optional, Sequence, cast
if TYPE_CHECKING:
    from bertopic import BERTopic

def bertopic_topic_words(model: "BERTopic")->tuple[tuple[str, ...], ...]:
  topic_probabilities = cast(
    Sequence[Sequence[tuple[str, float]]],
    tuple(model.get_topics().values())
    [model._outliers:]
  )
  topic_words = tuple(map(
    lambda distribution: tuple(map(
      lambda el: el[0],
      distribution
    )),
    topic_probabilities
  ))

  return topic_words

def bertopic_topic_labels(model: "BERTopic")->list[str]:
  topic_words = bertopic_topic_words(model)
  topic_labels = list(map(lambda words: ', '.join(filter(bool, words[:5])), topic_words))
  custom_labels = cast(Optional[list[str]], model.custom_labels_)
  labels: list[str]
  if custom_labels is not None:
    labels = list(map(lambda a, b: b or a, topic_labels, custom_labels[model._outliers:]))
  else:
    labels = topic_labels
  return labels

def bertopic_vectorize(model: "BERTopic", documents: Sequence[str])->tuple[Sequence[Sequence[str]], Sequence[str]]:
  vectorizer = model.vectorizer_model
  analyzer = vectorizer.build_analyzer()
  words = cast(Sequence[str], vectorizer.get_feature_names_out())
  tokens = [analyzer(doc) for doc in documents]
  return tokens, words