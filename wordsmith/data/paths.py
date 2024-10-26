import os
import shutil
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pandas as pd
import pydantic

from common.logger import RegisteredLogger
from common.models.api import ApiError

if TYPE_CHECKING:
  import bertopic
  import gensim

INTERMEDIATE_DIRECTORY = "intermediate"
RESULTS_DIRECTORY = "results"
DATA_DIRECTORY = "data"

class ProjectPaths(SimpleNamespace):
  Workspace = "workspace.parquet"
  Config = "config.json"

  Doc2Vec = "doc2vec"
  BERTopic = 'bertopic'

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

  @file_loading_error_handler("workspace table")
  def load_workspace(self)->pd.DataFrame:
    path = self.full_path(ProjectPaths.Workspace)
    return pd.read_parquet(path)
      
  @file_loading_error_handler("document embeddings")
  def load_doc2vec(self, column: str)->"gensim.models.Doc2Vec":
    import gensim
    path = self.assert_path(os.path.join(ProjectPaths.Doc2Vec, f"{column}.npy"))
    return cast(gensim.models.Doc2Vec, gensim.models.Doc2Vec.load(path))

  @file_loading_error_handler("topic information")
  def load_bertopic(self, column: str)->"bertopic.BERTopic":
    import bertopic
    path = self.assert_path(os.path.join(ProjectPaths.BERTopic, column))
    return bertopic.BERTopic.load(path)
  
  def cleanup(self):
    EXCLUDED = set(ProjectPaths.Config)
    for fnode in os.scandir(self.project_path):
      if fnode.name in EXCLUDED:
        continue
      if fnode.is_dir():
        shutil.rmtree(fnode.path)
      else:
        os.remove(fnode.path)