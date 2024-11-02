from typing import TYPE_CHECKING, Optional, cast
if TYPE_CHECKING:
    from bertopic import BERTopic

def bertopic_topic_labels(model: BERTopic, *, outliers: bool = False)->list[str]:
  topic_labels = model.generate_topic_labels(topic_prefix=False, separator=", ")
  custom_labels = cast(Optional[list[str]], model.custom_labels_)
  labels: list[str]
  if custom_labels is not None:
    labels = list(map(lambda a, b: b or a, topic_labels, custom_labels))
  else:
    labels = topic_labels
  if not outliers:
    labels = labels[model._outliers:]
  return labels