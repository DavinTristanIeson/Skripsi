from functools import lru_cache
from typing import Annotated, Any, ClassVar, Mapping, Union, cast
import pandas as pd
import pydantic
import json
import os

from common.logger import RegisteredLogger
from common.utils.loader import hashfile
from common.logger import TimeLogger

from wordsmith.data.schema import SchemaColumn, SchemaColumnType
from wordsmith.data.schema_manager import SchemaManager
from wordsmith.data.source import DataSource
from wordsmith.data.paths import DATA_DIRECTORY, ProjectPathManager, ProjectPaths

    
@pydantic.field_validator("columns", mode="before")
def __create_columns_field(cls, value):
  return SchemaManager.model_validate(dict(
    columns=value
  ))

SchemaManagerField = Annotated[SchemaManager, __create_columns_field]
  
logger = RegisteredLogger().provision("Config")
class Config(pydantic.BaseModel):
  project_id: str
  source: DataSource
  # schema is taken by pydantic
  dfschema: SchemaManagerField
  paths: ProjectPathManager

  @staticmethod
  def from_project(project_id: str)->"Config":
    source = os.path.join(DATA_DIRECTORY, project_id)
    with open(source, 'r', encoding='utf-8') as f:
      contents = json.load(f)
      contents["paths"] = ProjectPathManager(project_id=project_id)
      return Config.model_validate(contents)

  def preprocess(self):
    df = self.source.load()
    df = self.dfschema.preprocess(df)

    result_path = self.paths.full_path(ProjectPaths.Workspace)
    logger.info(f"Saving intermediate results to {result_path}")
    df.to_parquet(result_path)
    return df


__all__ = [
  "Config",
]
