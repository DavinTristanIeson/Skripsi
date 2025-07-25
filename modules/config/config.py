from typing import Any, Optional
import pandas as pd
import pydantic
import os

from modules.config.context import ConfigSerializationContext
from modules.exceptions.dataframe import DataFrameLoadException
from modules.exceptions.files import FileNotExistsException
from modules.logger import ProvisionedLogger
from modules.api import ApiError
from modules.storage.atomic import atomic_write

from .schema import SchemaManager
from .source import DataSource
from ..project.paths import DATA_DIRECTORY, ProjectPathManager, ProjectPaths
  
logger = ProvisionedLogger().provision("Config")
class ProjectMetadata(pydantic.BaseModel):
  name: str = pydantic.Field(min_length=1)
  description: Optional[str]
  tags: list[str]

class Config(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(use_enum_values=True)

  version: int = pydantic.Field(default=1)
  project_id: str
  metadata: ProjectMetadata
  source: DataSource
  # schema is taken by pydantic
  data_schema: SchemaManager

  @property
  def paths(self):
    return ProjectPathManager(project_id=self.project_id)
  
  @staticmethod
  def from_project(project_id: str)->"Config":
    import json

    data_directory = os.path.join(os.getcwd(), DATA_DIRECTORY)
    source_path = os.path.join(data_directory, project_id, "config.json")
    FileNotExistsException.verify(
      source_path,
      error=f"Project with ID {project_id} doesn't exist in {data_directory}. That project may not have been created yet, please create a new project first; or if you directly entered the project ID in the URL, please make sure that it is correctly spelled."
    )
    with open(source_path, 'r', encoding='utf-8') as f:
      contents = json.load(f)
      return Config.model_validate(contents)

  def save_to_json(self):
    import json

    config_path = ProjectPathManager(project_id=self.project_id).full_path(ProjectPaths.Config)
    logger.info(f"Saving config file in \"{config_path}\"")
    with atomic_write(config_path, mode="text") as f:
      json.dump(self.model_dump(context=ConfigSerializationContext(is_save=True)), f, indent=4, ensure_ascii=False)
    return
  
  def load_workspace(self)->pd.DataFrame:
    # Fix fastparquet limitation wherein it doesn't preserve ordered flag.
    path = self.paths.assert_path(ProjectPaths.Workspace)
    try:
      df = pd.read_parquet(path)
    except Exception as e:
      logger.error(e)
      raise DataFrameLoadException(
        message=DataFrameLoadException.format_message(
          path=path,
          purpose="workspace table",
          solution="Please load the data source again to recreate the workspace table. If this problem persists, consider resetting the environment and executing the topic modeling procedure again.",
          error=e
        )
      )
    for col in self.data_schema.ordered_categorical():
      if col.name not in df.columns:
        logger.warning(f"Failed to load {col.name} as it doesn't exist in the workspace. (Full column: {col})")
        continue
      # Set ordered categories
      col.fit(df) # type: ignore
    for col in self.data_schema.topic():
      if col.name in df.columns:
        # Force topic to be int32. Don't trust floats.
        col.fit(df) # type: ignore
    return df
  
  def save_workspace(self, df: pd.DataFrame):
    workspace_path = self.paths.full_path(ProjectPaths.Workspace)
    logger.info(f"Saving workspace file in \"{workspace_path}\"")
    with atomic_write(workspace_path, mode="binary") as f:
      df.to_parquet(f)
      
__all__ = [
  "Config",
]
