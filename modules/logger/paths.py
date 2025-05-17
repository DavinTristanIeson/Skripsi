import os
from types import SimpleNamespace

from modules.storage.paths import AbstractPathManager

class LogPaths(SimpleNamespace):
  ServerLogs = "server.log"
  TopicModelingLogs = "topic_modeling.log"
  TopicEvaluationLogs = "topic_evaluation.log"
  TopicModelExperimentLogs = "topic_model_experiments.log"

class LogPathManager(AbstractPathManager):
  @property
  def base_path(self)->str:
    return os.path.abspath("logs")