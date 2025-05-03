from typing import TYPE_CHECKING, Sequence, cast

import numpy as np
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.project.cache import ProjectCache
from modules.task.storage import TaskStorageProxy
from modules.topic.bertopic_ext.interpret import BERTopicInterpreter
from modules.topic.evaluation.method import cv_coherence, topic_diversity
from modules.topic.evaluation.model import BERTopicEvaluationPayload, BERTopicEvaluationResult, CoherenceVPerTopic
from modules.topic.model import Topic

if TYPE_CHECKING:
  from bertopic import BERTopic

def evaluate_topics(task: TaskStorageProxy, raw_documents: Sequence[str], topics: list[Topic], bertopic_model: "BERTopic"):
  topic_words = list(map(lambda topic: list(map(lambda word: word[0], topic.words)), topics))
  interpreter = BERTopicInterpreter(bertopic_model)
  documents = list(interpreter.tokenize(cast(Sequence[str], raw_documents)))

  cv_score, cv_scores_per_topic_raw = cv_coherence(topic_words, documents)
  cv_scores_per_topic_nparray = np.array(cv_scores_per_topic_raw)

  cv_scores_per_topic = list(map(
    lambda topic, coherence, std, support: CoherenceVPerTopic(topic=topic, coherence=coherence, std_dev=std, support=support),
    topics, cv_scores_per_topic_nparray[:, 0], cv_scores_per_topic_nparray[:, 1], cv_scores_per_topic_nparray[:, 2]
  ))

  diversity = topic_diversity(topic_words)

  return BERTopicEvaluationResult(
    coherence_v=cv_score,
    topic_diversity=diversity,
    coherence_v_per_topic=cv_scores_per_topic,
  )

def evaluate_topics_controller(task: TaskStorageProxy, cache: ProjectCache, payload: BERTopicEvaluationPayload):
  config = cache.config
  df = cache.load_workspace()
  column = cast(TextualSchemaColumn, config.data_schema.assert_of_type(payload.column, [SchemaColumnTypeEnum.Textual]))
  
  task.log_pending(f"Loading cached documents and topics for {column.name}")
  raw_documents = df[column.preprocess_column]
  mask = raw_documents.notna() & raw_documents != ''
  raw_documents = raw_documents[mask]

  tm_result = cache.load_topic(column.name)
  bertopic_model = cache.load_bertopic(column.name)
  task.log_success(f"Successfully loaded cached documents and topics for {column.name}.")

  task.log_pending(f"Evaluating the topics...")
  result = evaluate_topics(task, cast(Sequence[str], raw_documents), tm_result.topics, bertopic_model)
  task.log_success("Finished evaluating the topics.")

  task.success(result)

__all__ = [
  "evaluate_topics"
]