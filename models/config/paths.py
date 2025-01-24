import os
import shutil
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
import pydantic

from common.logger import RegisteredLogger
from common.models.api import ApiError
from ..topic.evaluation import ProjectTopicsEvaluationResult

if TYPE_CHECKING:
  import bertopic

INTERMEDIATE_DIRECTORY = "intermediate"
RESULTS_DIRECTORY = "results"
DATA_DIRECTORY = "data"

class ProjectPaths(SimpleNamespace):
  Config = "config.json"
  
  Workspace = "workspace.parquet"
  Topics = "topics.parquet"
  TopicHierarchy = "topic-hierarchy.parquet"
  Metadata = 'metadata'

  Embeddings = "embeddings"
  BERTopic = 'bertopic'
  Evaluation = "evaluation.json"

logger = RegisteredLogger().provision("Wordsmith Data Loader")

def file_loading_error_handler(entity_type: str):
  def decorator(fn):
    def inner(*args, **kwargs):
      try:
        return fn(*args, **kwargs)
      except ApiError as e:
        # Ignore api errors
        raise e
      except Exception as e:
        logger.error(f"Failed to load the {entity_type}. Error: {e}")
        raise ApiError(f"Failed to load the {entity_type}. Please wait for the topic modeling procedure to finish. If this problem persists, consider resetting the environment and executing the topic modeling procedure again.", 404)
    return inner
  return decorator

class ProjectPathManager(pydantic.BaseModel):
  project_id: str

  @property
  def project_path(self):
    project_dir = os.path.join(os.getcwd(), DATA_DIRECTORY, self.project_id)
    if not os.path.exists(project_dir):
      raise ApiError(f"No project exists with name: {self.project_id}.", 404)
    return project_dir
  
  def full_path(self, path: str)->str:
    project_dir = self.project_path
    fullpath:str = os.path.join(os.getcwd(), project_dir, path)
    return fullpath
  
  def assert_path(self, path: str)->str:
    path = self.full_path(path)
    if not os.path.exists(path):
      raise ApiError(f"{path} does not exist. Perhaps the file has not been created yet.", 404)
    return path

  @property
  def config_path(self):
    return self.full_path(ProjectPaths.Config)

  @file_loading_error_handler("workspace table")
  def load_workspace(self)->pd.DataFrame:
    path = self.full_path(ProjectPaths.Workspace)
    return pd.read_parquet(path)
      
  @file_loading_error_handler("document embeddings")
  def load_embeddings(self, column: str)->npt.NDArray:
    path = self.assert_path(os.path.join(ProjectPaths.Embeddings, f"{column}.npy"))
    return np.load(path)

  @file_loading_error_handler("BERTopic model")
  def load_bertopic(self, column: str)->"bertopic.BERTopic":
    import bertopic
    path = self.assert_path(os.path.join(ProjectPaths.BERTopic, f"{column}"))
    return bertopic.BERTopic.load(path)
  
  @file_loading_error_handler("topic information")
  def load_topics(self)->pd.DataFrame:
    path = self.assert_path(ProjectPaths.Topics)
    return pd.read_parquet(path)
  
  @file_loading_error_handler("topic hierarchy information")
  def load_topic_hierarchy(self)->pd.DataFrame:
    path = self.assert_path(ProjectPaths.TopicHierarchy)
    return pd.read_parquet(path)
  
  @file_loading_error_handler("additional information for rows")
  def load_row_metadata(self)->pd.DataFrame:
    path = self.assert_path(ProjectPaths.Metadata)
    return pd.read_parquet(path)
  
  def load_evaluation(self, column: str)->ProjectTopicsEvaluationResult:
    import json

    topics_path = self.full_path(os.path.join(ProjectPaths.BERTopic, f"{column}"))
    if not os.path.exists(topics_path):
      raise ApiError(f"The topic modeling procedure has not been run on {column} so we cannot determine the quality of the topics.", 400)
    path = self.assert_path(os.path.join(ProjectPaths.Evaluation))
    if not os.path.exists(path):
      raise ApiError(f"The quality of the topics discovered in {column} has never been evaluated before. Please manually start the topic evaluation procedure.", 400)
    
    try:
      with open(path) as f:
        return ProjectTopicsEvaluationResult.model_validate(json.load(f))
    except Exception as e:
      logger.error(f"Failed to load evaluation data. Error => {e}")
      raise ApiError(f"An unexpected error has occurred while loading evaluation data from {ProjectPaths.Evaluation}. The file may have been corrupted; try to delete the file and then execute the topic evaluation procedure again.", 500)
  
  def cleanup(self, all: bool = False):
    directories = [
      self.full_path(ProjectPaths.Embeddings),
      self.full_path(ProjectPaths.BERTopic),
    ]
    files = [
      self.full_path(ProjectPaths.Workspace),
      self.full_path(ProjectPaths.Evaluation),
      self.full_path(ProjectPaths.Topics),
      self.full_path(ProjectPaths.TopicHierarchy),
      self.full_path(ProjectPaths.Metadata),
    ]

    if all:
      files.extend([
        self.full_path(ProjectPaths.Config),
      ])

    try:
      for dir in directories:
        if os.path.exists(dir):
          shutil.rmtree(dir)
      for file in files:
        if os.path.exists(file):
          os.remove(file)
    except Exception as e:
      logger.error(f"An error has occurred while deleting directories and/or files from the project directory of {self.project_id}. Error => {e}")
      raise ApiError(f"An unexpected error has occurred while cleaning up the project directory of {self.project_id}: {e}", 500)
    
    if all and os.path.exists(self.project_path):
      remaining_files = os.listdir(self.project_path)
      if len(remaining_files) == 0:
        try:
          os.rmdir(self.project_path)
        except ApiError as e:
          logger.error(f"An error has occurred while deleting {self.project_path}. Error => {e}")
          raise ApiError(f"An unexpected error has occurred while cleaning up the project directory of {self.project_id}: {e}", 500)
      else:
        logger.warn(f"Skipping the deletion of {self.project_path} as there are non-wordsmith-managed files in the folder: {remaining_files}")

