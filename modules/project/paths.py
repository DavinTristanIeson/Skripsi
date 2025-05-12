import os
from types import SimpleNamespace
from typing import Optional

import pydantic

from modules.logger import ProvisionedLogger
from modules.api import ApiError
from modules.storage.paths import AbstractPathManager

INTERMEDIATE_DIRECTORY = "intermediate"
RESULTS_DIRECTORY = "results"
DATA_DIRECTORY = "data"

class ProjectPaths(SimpleNamespace):
  Config = "config.json"
  
  Workspace = "workspace.parquet"

  TopicModelingFolderName = "topic-modeling"
  UserDataFolder = "userdata"
  @staticmethod
  def UserData(type: str):
    return os.path.join(ProjectPaths.UserDataFolder, f"{type}.json")

  # Column Specific
  @staticmethod
  def Column(name: str)->str:
    import base64
    encoded_name = base64.b64encode(name.encode('utf-8'))
    return encoded_name.decode('utf-8')
  
  @staticmethod
  def TopicModelingFolder(name: str)->str:
    return os.path.join(ProjectPaths.TopicModelingFolderName, name)
  
  TopicsFileName = "topics.json"
  @staticmethod
  def Topics(column: str):
    return os.path.join(ProjectPaths.TopicModelingFolder(column), ProjectPaths.TopicsFileName)

  DocumentEmbeddingsFileName = "document_vectors.npy"
  @staticmethod
  def DocumentEmbeddings(column: str):
    return os.path.join(
      ProjectPaths.TopicModelingFolder(column),
      ProjectPaths.DocumentEmbeddingsFileName
    )

  UMAPEmbeddingsFileName = "umap_vectors.npy"
  @staticmethod
  def UMAPEmbeddings(column: str):
    return os.path.join(
      ProjectPaths.TopicModelingFolder(column), 
      ProjectPaths.UMAPEmbeddingsFileName    
    )
  
  VisualizationEmbeddingsFileName = "visualization_vectors.npy"
  @staticmethod
  def VisualizationEmbeddings(column: str):
    return os.path.join(
      ProjectPaths.TopicModelingFolder(column),
      ProjectPaths.VisualizationEmbeddingsFileName
    )
  
  @staticmethod
  def EmbeddingModel(column: str, model: str):
    return os.path.join(ProjectPaths.TopicModelingFolder(column), model)

  BERTopicFolder = 'bertopic'

  @staticmethod
  def BERTopic(column: str):
    return os.path.join(ProjectPaths.TopicModelingFolder(column), ProjectPaths.BERTopicFolder)
  
  TopicModelExperimentsFileName = "topic_model_experiments.json"
  @staticmethod
  def TopicModelExperiments(column: str):
    return os.path.join(ProjectPaths.TopicModelingFolder(column), ProjectPaths.TopicModelExperimentsFileName)
  
  TopicEvaluationFileName = "topic_evaluation.json"
  @staticmethod
  def TopicEvaluation(column: str):
    return os.path.join(ProjectPaths.TopicModelingFolder(column), ProjectPaths.TopicEvaluationFileName)
  
  # Logs
  LogsFolder = "logs"

  @staticmethod
  def ColumnLogsFolder(column: str):
    return os.path.join(ProjectPaths.TopicModelingFolder(column), ProjectPaths.LogsFolder)

  @staticmethod
  def TopicModelingLogs(column: str):
    return os.path.join(ProjectPaths.ColumnLogsFolder(column), "topic_modeling.log")
  
  @staticmethod
  def TopicEvaluationLogs(column: str):
    return os.path.join(ProjectPaths.ColumnLogsFolder(column), "topic_evaluation.log")
  
  @staticmethod
  def TopicModelExperimentLogs(column: str):
    return os.path.join(ProjectPaths.ColumnLogsFolder(column), "topic_model_experiments.log")

logger = ProvisionedLogger().provision("Wordsmith Data Loader")

class ProjectPathManager(pydantic.BaseModel, AbstractPathManager):
  project_id: str
  
  @property
  def base_path(self):
    project_dir = os.path.join(os.getcwd(), DATA_DIRECTORY, self.project_id)
    return project_dir

  @property
  def project_path(self):
    return self.base_path
  
  def assert_path(self, path: str)->str:
    if not os.path.exists(self.base_path):
      raise ApiError(f"No project exists with ID: {self.project_id}.", 404)
    return super().assert_path(path)

  @property
  def config_path(self):
    return self.full_path(ProjectPaths.Config)

  def cleanup(self, all: bool = False):
    directories = [
      ProjectPaths.TopicModelingFolderName,
    ]
    files = []

    if all:
      files.extend([
        ProjectPaths.UserDataFolder,
        ProjectPaths.Config,
      ])
    self._cleanup(directories, files, soft=not all)

  def cleanup_topic_modeling(self, column: str):
    directories = [
      ProjectPaths.TopicModelingFolder(column),
    ]
    files = []
    self._cleanup(directories, files, soft=True)

__all__ = [
  "ProjectPathManager",
  "ProjectPaths",
  "DATA_DIRECTORY"
]