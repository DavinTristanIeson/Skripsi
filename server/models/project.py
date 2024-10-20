import http
import os
from typing import Sequence

import pydantic
from common.models.api import ApiError
from wordsmith.data.config import Config
from wordsmith.data.schema import SchemaColumnType
from wordsmith.data.source import DataSource

# Resource
class ProjectResource(pydantic.BaseModel):
  id: str
  config: Config

class CheckProjectIdResource(pydantic.BaseModel):
  available: bool

class DatasetInferredColumnResource(pydantic.BaseModel):
  # Configurations that FE can use to autofill schema.
  name: str
  type: SchemaColumnType

class CheckDatasetResource(pydantic.BaseModel):
  columns: Sequence[DatasetInferredColumnResource]


# Schema
class CheckProjectIdSchema(pydantic.BaseModel):
  project_id: str

class CheckDatasetSchema(pydantic.RootModel):
  root: DataSource

  @pydantic.field_validator('root', mode="after")
  def validate_file_path(cls, root: DataSource):
    if not os.path.exists(root.path):
      raise ApiError(f"Cannot find any file at {root.path}. Are you sure you have provided the correct path?", http.HTTPStatus.NOT_FOUND)
    
    if not os.path.isfile(root.path):
      raise ApiError(f"The item at {root.path} is not a file. Are you sure you have provided the correct path?", http.HTTPStatus.BAD_REQUEST)
    
    return root