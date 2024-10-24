from enum import Enum
import http
import os
from typing import Generic, Optional, Sequence, TypeVar

import pydantic
from common.ipc.responses import IPCResponse
from common.models.api import ApiError
from common.models.enum import EnumMemberDescriptor, ExposedEnum
from wordsmith.data.config import Config, ProjectIdField
from wordsmith.data.schema import SchemaColumnTypeEnum
from wordsmith.data.source import DataSource

# Common resources

class ProjectTaskStatus(str, Enum):
  Idle = "idle"
  Pending = "pending"
  Success = "success"
  Failed = "failed"

ExposedEnum().register(ProjectTaskStatus, {
  ProjectTaskStatus.Idle: EnumMemberDescriptor("Idle"),
  ProjectTaskStatus.Pending: EnumMemberDescriptor("Pending"),
  ProjectTaskStatus.Success: EnumMemberDescriptor("Success"),
  ProjectTaskStatus.Failed: EnumMemberDescriptor("Failed"),
})

T = TypeVar("T")
class ProjectTaskResult(pydantic.BaseModel, Generic[T]):
  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

  data: T
  status: ProjectTaskStatus
  message: Optional[str]
  progress: Optional[float]
  error: Optional[str]

  @staticmethod
  def from_ipc(response: IPCResponse):
    return ProjectTaskResult.model_validate(response.model_dump())



# Resource
class ProjectLiteResource(pydantic.BaseModel):
  # This resource doesn't have any other fields for now, and probably for the foreseeable future.
  # But we're making it a resource anyway in case a new feature introduces a new field to this resource.
  id: str
  path: str

class ProjectResource(pydantic.BaseModel):
  id: str
  config: Config

class CheckProjectIdResource(pydantic.BaseModel):
  available: bool

class DatasetInferredColumnResource(pydantic.BaseModel):
  # Configurations that FE can use to autofill schema.
  name: str
  type: SchemaColumnTypeEnum

class CheckDatasetResource(pydantic.BaseModel):
  columns: Sequence[DatasetInferredColumnResource]


# Schema
class CheckProjectIdSchema(pydantic.BaseModel):
  project_id: ProjectIdField

class CheckDatasetSchema(pydantic.RootModel):
  root: DataSource

  @pydantic.field_validator('root', mode="after")
  def validate_file_path(cls, root: DataSource):
    if not os.path.exists(root.path):
      raise ApiError(f"Cannot find any file at {root.path}. Are you sure you have provided the correct path?", http.HTTPStatus.NOT_FOUND)
    
    if not os.path.isfile(root.path):
      raise ApiError(f"The item at {root.path} is not a file. Are you sure you have provided the correct path?", http.HTTPStatus.BAD_REQUEST)
    
    return root