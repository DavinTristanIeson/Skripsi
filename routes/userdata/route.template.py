from http import HTTPStatus
from typing import Annotated, Any
from fastapi import APIRouter, Depends

from routes.dependencies.project import ProjectCacheDependency
from modules.storage.userdata import (
  UserDataResource, UserDataSchema, UserDataStorageController
)
from modules.api.wrapper import ApiError, ApiResult
from modules.dashboard import Dashboard
from modules.project.paths import ProjectPaths
from modules.table.filter_variants import TableFilter

from .model import ComparisonState

router = APIRouter(
  tags=["Storage"]
)

def filter_validator(data: Any)->UserDataResource[TableFilter]:
  return UserDataResource[TableFilter].model_validate(data)

def comparison_state_validator(model: Any)->UserDataResource[ComparisonState]:
  return UserDataResource[ComparisonState].model_validate(model)

def dashboard_validator(model: Any)->UserDataResource[Dashboard]:
  return UserDataResource[Dashboard].model_validate(model)

#CODEGEN SPLIT#

#====================#
# CODEGEN_PASCALNAME
#====================#

def get_CODEGEN_NAME_storage_controller(cache: ProjectCacheDependency):
  return UserDataStorageController[CODEGEN_CLASSNAME](
    path=cache.config.paths.full_path(ProjectPaths.UserData("CODEGEN_NAME")),
    validator=CODEGEN_VALIDATOR,
  )

CODEGEN_PASCALNAMEStorageDependency = Annotated[UserDataStorageController, Depends(get_CODEGEN_NAME_storage_controller)]

@router.get("/CODEGEN_URL")
def get__all_CODEGEN_NAME(storage: CODEGEN_PASCALNAMEStorageDependency)->ApiResult[list[UserDataResource[CODEGEN_CLASSNAME]]]:
  return ApiResult(
    data=storage.all(),
    message=None
  )

@router.get("/CODEGEN_URL/{id}")
def get__CODEGEN_NAME(storage: CODEGEN_PASCALNAMEStorageDependency, id: str)->ApiResult[UserDataResource[CODEGEN_CLASSNAME]]:
  data = storage.get(id)
  if data is None:
    raise ApiError(f"We were not able to find any CODEGEN_LABEL with ID \"{id}\".", HTTPStatus.NOT_FOUND)
  return ApiResult(
    data=data,
    message=None
  )

@router.post("/CODEGEN_URL")
def post__CODEGEN_NAME(storage: CODEGEN_PASCALNAMEStorageDependency, body: UserDataSchema[CODEGEN_CLASSNAME])->ApiResult[None]:
  storage.create(body)
  return ApiResult(data=None, message="The CODEGEN_LABEL has been successfully created.")

@router.put("/CODEGEN_URL/{id}")
def put__CODEGEN_NAME(storage: CODEGEN_PASCALNAMEStorageDependency, id: str, body: UserDataSchema[CODEGEN_CLASSNAME])->ApiResult[None]:
  storage.update(id, body)
  return ApiResult(data=None, message="The CODEGEN_LABEL has been successfully updated.")
  
@router.delete("/CODEGEN_URL/{id}")
def delete__CODEGEN_NAME(storage: CODEGEN_PASCALNAMEStorageDependency, id: str)->ApiResult[None]:
  storage.delete(id)
  return ApiResult(data=None, message="The CODEGEN_LABEL has been successfully removed.")
