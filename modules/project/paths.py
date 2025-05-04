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
  TopicsFolder = "topics"
  UserDataFolder = "userdata"

  @staticmethod
  def hash_name(name: str)->str:
    import base64
    encoded_name = base64.b64encode(name.encode('utf-8'))
    return encoded_name.decode('utf-8')

  @staticmethod
  def Topics(column: str):
    return os.path.join(ProjectPaths.TopicsFolder, f"{ProjectPaths.hash_name(column)}.json")

  EmbeddingsFolder = "embedding"

  @staticmethod
  def DocumentEmbeddings(column: str):
    return os.path.join(
      ProjectPaths.EmbeddingsFolder,
      ProjectPaths.hash_name(column),
      "document_vectors.npy"
    )

  @staticmethod
  def UMAPEmbeddings(column: str):
    return os.path.join(
      ProjectPaths.EmbeddingsFolder,
      ProjectPaths.hash_name(column), 
      "umap_embeddings.npy"
    )
  
  @staticmethod
  def VisualizationEmbeddings(column: str):
    return os.path.join(
      ProjectPaths.EmbeddingsFolder,
      ProjectPaths.hash_name(column),
      "visualization_embeddings.npy"
    )
  
  @staticmethod
  def EmbeddingModel(column: str, model: Optional[str]):
    prefix = os.path.join(
      ProjectPaths.EmbeddingsFolder,
      ProjectPaths.hash_name(column)
    )
    if model is None:
      return prefix
    return os.path.join(prefix, model)

  BERTopicFolder = 'bertopic'

  @staticmethod
  def BERTopic(column: str):
    return os.path.join(ProjectPaths.BERTopicFolder, ProjectPaths.hash_name(column))
  
  EvaluationFolder = 'evaluation'
  @staticmethod
  def TopicExperiments(column: str):
    return os.path.join(ProjectPaths.EvaluationFolder, f"topic_experiment_{ProjectPaths.hash_name(column)}.json")
  
  @staticmethod
  def TopicEvaluation(column: str):
    return os.path.join(ProjectPaths.EvaluationFolder, f"topic_evaluation_{ProjectPaths.hash_name(column)}.json")

  @staticmethod
  def TopicModelingPaths(column: str):
    return [
      ProjectPaths.BERTopic(column),
      ProjectPaths.DocumentEmbeddings(column),
      ProjectPaths.VisualizationEmbeddings(column),
      ProjectPaths.UMAPEmbeddings(column),
      ProjectPaths.Topics(column),
      ProjectPaths.TopicExperiments(column),
      ProjectPaths.TopicEvaluation(column),
    ]
  
  @staticmethod
  def UserData(type: str):
    return os.path.join(ProjectPaths.UserDataFolder, f"{type}.json")

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
      ProjectPaths.EmbeddingsFolder,
      ProjectPaths.BERTopicFolder,
      ProjectPaths.TopicsFolder,
    ]
    files = [
      ProjectPaths.Workspace,
    ]

    if all:
      files.extend([
        ProjectPaths.UserDataFolder,
        ProjectPaths.Config,
      ])
    self._cleanup(directories, files)

  def cleanup_topic_modeling(self):
    directories = [
      ProjectPaths.EmbeddingsFolder,
      ProjectPaths.BERTopicFolder,
      ProjectPaths.TopicsFolder,
      ProjectPaths.EvaluationFolder,
    ]
    files = []

    self._cleanup(directories, files)

  def cleanup_topic_experiments(self):
    self._cleanup(
      directories=[ProjectPaths.EvaluationFolder],
      files=[]
    )

__all__ = [
  "ProjectPathManager",
  "ProjectPaths",
  "DATA_DIRECTORY"
]