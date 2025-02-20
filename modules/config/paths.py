import os
import shutil
from types import SimpleNamespace

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

  @staticmethod
  def Topics(column: str):
    return os.path.join(ProjectPaths.TopicsFolder, f"{column}.json")

  EmbeddingsFolder = "embeddings"

  @staticmethod
  def DocumentEmbeddings(column: str):
    return os.path.join(ProjectPaths.EmbeddingsFolder, column, "document_embeddings.npy")

  @staticmethod
  def UMAPEmbeddings(column: str):
    return os.path.join(ProjectPaths.EmbeddingsFolder, column, "umap_embeddings.npy")
  
  @staticmethod
  def VisualizationEmbeddings(column: str):
    return os.path.join(ProjectPaths.EmbeddingsFolder, column, "visualization_embeddings.npy")
  
  @staticmethod
  def EmbeddingModel(column: str, model: str):
    return os.path.join(ProjectPaths.EmbeddingsFolder, column, model)

  BERTopicFolder = 'bertopic'

  @staticmethod
  def BERTopic(column: str):
    return os.path.join(ProjectPaths.BERTopicFolder, column)

logger = ProvisionedLogger().provision("Wordsmith Data Loader")

class ProjectPathManager(pydantic.BaseModel, AbstractPathManager):
  project_id: str

  @property
  def base_path(self):
    project_dir = os.path.join(os.getcwd(), DATA_DIRECTORY, self.project_id)
    if not os.path.exists(project_dir):
      raise ApiError(f"No project exists with name: {self.project_id}.", 404)
    return project_dir

  @property
  def project_path(self):
    return self.base_path

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
        ProjectPaths.Config,
      ])
    self._cleanup(directories, files)

  def cleanup_topic_modeling(self):
    directories = [
      ProjectPaths.EmbeddingsFolder,
      ProjectPaths.BERTopicFolder,
      ProjectPaths.TopicsFolder,
    ]
    files = []

    self._cleanup(directories, files)

__all__ = [
  "ProjectPathManager",
  "ProjectPaths",
  "DATA_DIRECTORY"
]