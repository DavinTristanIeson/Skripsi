from typing import Annotated, Any, Callable, Optional, cast
import pydantic
import json
import os

from common.logger import RegisteredLogger

from common.models.api import ApiError
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
    current: dict[str, Any] = cast(dict[str, Any], self)
    if "project_id" in current and isinstance(current["project_id"], str):
      current["paths"] = ProjectPathManager(project_id=current["project_id"])
    return current

  @staticmethod
  def from_project(project_id: str)->"Config":
    data_directory = os.path.join(os.getcwd(), DATA_DIRECTORY)
    source = os.path.join(data_directory, project_id, "config.json")
    if not os.path.exists(source):
      raise ApiError(f"Project with ID {project_id} doesn't exist in {data_directory}. That project may not have been created yet, please create a new project first; or if you directly entered the project ID in the URL, please make sure that it is correctly spelled.", 404)
    with open(source, 'r', encoding='utf-8') as f:
      contents = json.load(f)
      return Config.model_validate(contents)

  def preprocess(self, *, on_start: Optional[Callable[[SchemaColumn], None]] = None):
    df = self.source.load()
    df = self.dfschema.preprocess(df, on_start=on_start)

    result_path = self.paths.full_path(ProjectPaths.Workspace)
    logger.info(f"Saving intermediate results to {result_path}")
    df.to_parquet(result_path)
    return df

  def save_to_json(self, folder_path: str):
    config_file = os.path.join(folder_path, "config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(self.model_dump(), f, indent=4, ensure_ascii=False)
    return

__all__ = [
  "Config",
]
