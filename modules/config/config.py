from typing import Any
import pandas as pd
import pydantic
import os

from modules.logger import ProvisionedLogger
from modules.api import ApiError
from modules.validation import FilenameField

from .schema_manager import SchemaManager
from .source import DataSource
from .paths import DATA_DIRECTORY, ProjectPathManager, ProjectPaths

  
logger = ProvisionedLogger().provision("Config")
class Config(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(use_enum_values=True)

  version: int = pydantic.Field(default=1)
  project_id: FilenameField
  source: DataSource
  # schema is taken by pydantic
  data_schema: SchemaManager

  paths: ProjectPathManager = pydantic.Field(exclude=True)
  
  @pydantic.model_validator(mode="before")
  @classmethod
  def __validate__paths(cls, current: dict[str, Any]):
    if "project_id" in current and isinstance(current["project_id"], str):
      current["paths"] = ProjectPathManager(project_id=current["project_id"])
    return current

  @staticmethod
  def from_project(project_id: str)->"Config":
    import json

    data_directory = os.path.join(os.getcwd(), DATA_DIRECTORY)
    source = os.path.join(data_directory, project_id, "config.json")
    if not os.path.exists(source):
      raise ApiError(f"Project with ID {project_id} doesn't exist in {data_directory}. That project may not have been created yet, please create a new project first; or if you directly entered the project ID in the URL, please make sure that it is correctly spelled.", 404)
    with open(source, 'r', encoding='utf-8') as f:
      contents = json.load(f)
      return Config.model_validate(contents)

  def save_to_json(self):
    import json

    config_file = os.path.join(self.paths.project_path, "config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
      json.dump(self.model_dump(), f, indent=4, ensure_ascii=False)
    return
  
  def load_workspace(self)->pd.DataFrame:
    # Fix fastparquet limitation wherein it doesn't preserve ordered flag.
    path = self.paths.full_path(ProjectPaths.Workspace)
    try:
      df = pd.read_parquet(path)
    except Exception as e:
      logger.error(e)
      raise ApiError(f"Failed to load the workspace table from {path}. Please load the data source again to recreate the workspace table. If this problem persists, consider resetting the environment and executing the topic modeling procedure again.", 404)
    for col in self.data_schema.ordered_categorical():
      # Set ordered categories
      col.fit(df) # type: ignore
    return df
      
__all__ = [
  "Config",
]
