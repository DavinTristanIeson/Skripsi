import os
import shutil
from types import SimpleNamespace

import pandas as pd
import pydantic

from common.logger import RegisteredLogger
from server.controllers.exceptions import ApiError

INTERMEDIATE_DIRECTORY = "intermediate"
RESULTS_DIRECTORY = "results"
DATA_DIRECTORY = "data"

class ProjectPaths(SimpleNamespace):
  Workspace = "workspace.parquet"
  Embeddings = "embeddings.npy"
  Config = "config.json"
  TopicsTable = 'topics.parquet'

  Doc2VecModel = "doc2vec"
  BERTopicModel = 'bertopic'

logger = RegisteredLogger().provision("Wordsmith Data Loader")

class ProjectPathManager(pydantic.BaseModel):
  project_id: str

  @property
  def project_path(self):
    project_dir = os.path.join(DATA_DIRECTORY, self.project_id)
    if not os.path.exists(project_dir):
      raise ApiError(f"No project exists with ID {self.project_id}.", 404)
    return project_dir
  
  def full_path(self, path: str)->str:
    project_dir = self.project_path
    fullpath:str = os.path.join(project_dir, path)
    if not os.path.exists(fullpath):
      raise ApiError(f"{fullpath} does not exist. Perhaps the file has not been created yet.", 404)
    return fullpath

  def load_preprocessed_table(self)->pd.DataFrame:
    path = self.full_path(ProjectPaths.Workspace)
    return pd.read_parquet(path)
  
  def load_topics_table(self)->pd.DataFrame:
    path = self.full_path(ProjectPaths.TopicsTable)
    return pd.read_parquet(path)
  
  def cleanup(self):
    EXCLUDED = set(ProjectPaths.Config)
    for fnode in os.scandir(self.project_path):
      if fnode.name in EXCLUDED:
        continue
      if fnode.is_dir():
        shutil.rmtree(fnode.path)
      else:
        os.remove(fnode.path)