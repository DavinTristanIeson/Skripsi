from enum import Enum
import logging
from typing import Any, ClassVar, Literal, Mapping, Optional, Union, cast
import pandas as pd
import pydantic
import json
import os

from wordsmith.data.cache import ConfigPathsManager
from wordsmith.utils.loader import hashfile

from .modules import *
from .source import *
from .schema import *
from .common import *

from wordsmith.debug.logger import TimeLogger

class Config(pydantic.BaseModel):
  name: str
  source: DataSource
  path: str
  columns: tuple[SchemaColumn, ...]
  paths: ConfigPathsManager
  metadata: dict[str, Any] = pydantic.Field(default_factory=lambda: {})
  
  def get_columns(self, type: SchemaColumnType)->tuple[SchemaColumn,...]:
    return tuple(filter(lambda x: x.type == type, self.columns))


  LOG_NAME: ClassVar[str] = "Config"

  @staticmethod
  def parse(source: Union[str, Mapping[str, Any]])->"Config":
    if isinstance(source, str):
      source = os.path.normpath(source)
      with open(source, 'r', encoding='utf-8') as f:
        contents = json.load(f)
        dirpath = os.path.dirname(source)
        contents["path"] = dirpath

        paths_manager = ConfigPathsManager(path=dirpath)
        contents["paths"] = paths_manager
        return Config.model_validate(contents, context={"paths": paths_manager})
    else:
      return Config.model_validate(source)
    

  def load(self, *, cache:bool = True, preprocessing:bool = True, show_progress:bool=True)->pd.DataFrame:
    df_path = self.paths.access_path(RelevantDataPaths.Table)
    hash_path = self.paths.access_path(RelevantDataPaths.DataSourceHash)
    intermediate_path = self.paths.access_path(RelevantDataPaths.IntermediateDirectory)
    config_path = self.paths.access_path(RelevantDataPaths.Config)
    logger = logging.getLogger(Config.LOG_NAME)
    if cache:
      config_hash = hashfile(config_path)
      source_hash = self.source.hash()

      if self.paths.can_reuse_cache(config_hash + source_hash):
        logger.info(f"Using intermediate results already cached in {df_path}")
        return pd.read_parquet(df_path)

    df = self.source.load()
    df = df.loc[:, [col.name for col in self.columns]]
    logger.info(f"Loaded data source from {self.source.path}")
    for col in self.columns:
      df.loc[:, col.name] = col.fit(cast(pd.Series, df.loc[:, col.name]))
      if col.type != SchemaColumnType.Text or not preprocessing:
        continue

      with TimeLogger(Config.LOG_NAME, f"Preprocessing {col.name}"):
        preprocess_column = col.get_preprocess_column()
        df[preprocess_column] = pd.Series(col.preprocessing.apply(df[col.name], show_progress=show_progress)) # type: ignore

    print(df)


    if cache:
      logger.info(f"Saving intermediate results to {intermediate_path}")
      if not os.path.exists(intermediate_path):
        os.mkdir(intermediate_path)
        logger.info(f"Created an intermediate folder: {intermediate_path}, as it hasn't existed before.")

      config_hash = hashfile(config_path)
      source_hash = self.source.hash()
      with open(hash_path, 'w', encoding='utf-8') as f:
        f.write(config_hash + source_hash)
      df.to_parquet(df_path)
    return df


__all__ = [
  "Config",
]
