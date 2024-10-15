from dataclasses import dataclass
import json
import os
from typing import Optional

import pandas as pd

from wordsmith.data.common import RelevantDataPaths
from wordsmith.data.source import DataSource

import gensim


@dataclass
class ConfigPathsManager:
  path: str

  def access_path(self, path: RelevantDataPaths)->str:
    return os.path.join(os.path.normpath(self.path), path)
    
  def can_reuse_cache(self, hash: str)->bool:
    hash_path = self.access_path(RelevantDataPaths.DataSourceHash)
    df_path = self.access_path(RelevantDataPaths.Table)

    if not os.path.exists(hash_path) or not os.path.exists(df_path):
      return False
    
    with open(hash_path, encoding='utf-8') as f:
      current_hash = f.read()

    return hash == current_hash
  
  def load_stemming_dictionary(self)->dict[str, str]:
    dictionary_path = self.access_path(RelevantDataPaths.StemmingDictionary)
    with open(dictionary_path, 'r', encoding='utf-8') as f:
      return json.load(f)

  def load_dictionary(self)->gensim.corpora.Dictionary:
    dictionary_path = self.access_path(RelevantDataPaths.GensimDictionary)
    return gensim.corpora.Dictionary.load(dictionary_path)
  
  def load_table(self)->pd.DataFrame:
    df_path = self.access_path(RelevantDataPaths.Table)
    return pd.read_parquet(df_path)
    
  def initialize_paths(self)->None:
    intermediate_dirpath = os.path.join(self.path, RelevantDataPaths.IntermediateDirectory)
    if not os.path.exists(intermediate_dirpath):
      os.mkdir(intermediate_dirpath)

    results_dirpath = os.path.join(self.path, RelevantDataPaths.ResultDirectory )
    if not os.path.exists(results_dirpath):
      os.mkdir(results_dirpath)
  
  def intermediate_path(self, path: str):
    dirpath = os.path.join(self.path, RelevantDataPaths.IntermediateDirectory)
    self.initialize_paths()
    return os.path.join(dirpath, os.path.normpath(path))
  
  def result_path(self, path: str):
    dirpath = os.path.join(self.path, RelevantDataPaths.ResultDirectory)
    self.initialize_paths()
    return os.path.join(dirpath, os.path.normpath(path))
  
  def load_result_table(self):
    path = os.path.join(self.path, RelevantDataPaths.ResultTable)
    return pd.read_parquet(path)
  
  def load_topics_table(self):
    path = os.path.join(self.path, RelevantDataPaths.Topics)
    return pd.read_parquet(path)
  
__all__ = [
  "ConfigPathsManager"
]