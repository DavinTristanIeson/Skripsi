from typing import Annotated, Callable, Literal, Optional
import pydantic
import json
import os

from common.logger import RegisteredLogger

from wordsmith.data.schema import SchemaColumn
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
  version: int = pydantic.Field(default=1)
  project_id: str
  source: DataSource
  # schema is taken by pydantic
  dfschema: SchemaManagerField
  paths: ProjectPathManager = pydantic.Field(exclude=True)
  
  @pydantic.model_validator(mode="before")
  def __validate__paths(self):
    if self.project_id is not None and isinstance(self.project_id, str):
      self.paths = ProjectPathManager(project_id=self.project_id)

  @staticmethod
  def from_project(project_id: str)->"Config":
    source = os.path.join(DATA_DIRECTORY, project_id, "config.json")
    with open(source, 'r', encoding='utf-8') as f:
      contents = json.load(f)
      contents["paths"] = ProjectPathManager(project_id=project_id)
      return Config.model_validate(contents)

  def preprocess(self, *, on_start: Optional[Callable[[SchemaColumn], None]] = None):
    df = self.source.load()
    df = self.dfschema.preprocess(df, on_start=on_start)

    result_path = self.paths.full_path(ProjectPaths.Workspace)
    logger.info(f"Saving intermediate results to {result_path}")
    df.to_parquet(result_path)
    return df


__all__ = [
  "Config",
]
