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



#====================#
# Filters
#====================#

def get_filters_storage_controller(cache: ProjectCacheDependency):
  return UserDataStorageController[TableFilter](
    path=cache.config.paths.full_path(ProjectPaths.UserData("filters")),
    validator=filter_validator,
  )

FiltersStorageDependency = Annotated[UserDataStorageController, Depends(get_filters_storage_controller)]

@router.get("/filters")
def get__all_filters(storage: FiltersStorageDependency)->ApiResult[list[UserDataResource[TableFilter]]]:
  return ApiResult(
    data=storage.all(),
    message=None
  )

@router.get("/filters/{id}")
def get__filters(storage: FiltersStorageDependency, id: str)->ApiResult[UserDataResource[TableFilter]]:
  data = storage.get(id)
  if data is None:
    raise ApiError(f"We were not able to find any filter with ID \"{id}\".", HTTPStatus.NOT_FOUND)
  return ApiResult(
    data=data,
    message=None
  )

@router.post("/filters")
def post__filters(storage: FiltersStorageDependency, body: UserDataSchema[TableFilter])->ApiResult[None]:
  storage.create(body)
  return ApiResult(data=None, message="The filter has been successfully created.")

@router.put("/filters/{id}")
def put__filters(storage: FiltersStorageDependency, id: str, body: UserDataSchema[TableFilter])->ApiResult[None]:
  storage.update(id, body)
  return ApiResult(data=None, message="The filter has been successfully updated.")
  
@router.delete("/filters/{id}")
def delete__filters(storage: FiltersStorageDependency, id: str)->ApiResult[None]:
  storage.delete(id)
  return ApiResult(data=None, message="The filter has been successfully removed.")


#====================#
# ComparisonState
#====================#

def get_comparison_state_storage_controller(cache: ProjectCacheDependency):
  return UserDataStorageController[ComparisonState](
    path=cache.config.paths.full_path(ProjectPaths.UserData("comparison_state")),
    validator=comparison_state_validator,
  )

ComparisonStateStorageDependency = Annotated[UserDataStorageController, Depends(get_comparison_state_storage_controller)]

@router.get("/comparison-state")
def get__all_comparison_state(storage: ComparisonStateStorageDependency)->ApiResult[list[UserDataResource[ComparisonState]]]:
  return ApiResult(
    data=storage.all(),
    message=None
  )

@router.get("/comparison-state/{id}")
def get__comparison_state(storage: ComparisonStateStorageDependency, id: str)->ApiResult[UserDataResource[ComparisonState]]:
  data = storage.get(id)
  if data is None:
    raise ApiError(f"We were not able to find any comparison groups with ID \"{id}\".", HTTPStatus.NOT_FOUND)
  return ApiResult(
    data=data,
    message=None
  )

@router.post("/comparison-state")
def post__comparison_state(storage: ComparisonStateStorageDependency, body: UserDataSchema[ComparisonState])->ApiResult[None]:
  storage.create(body)
  return ApiResult(data=None, message="The comparison groups has been successfully created.")

@router.put("/comparison-state/{id}")
def put__comparison_state(storage: ComparisonStateStorageDependency, id: str, body: UserDataSchema[ComparisonState])->ApiResult[None]:
  storage.update(id, body)
  return ApiResult(data=None, message="The comparison groups has been successfully updated.")
  
@router.delete("/comparison-state/{id}")
def delete__comparison_state(storage: ComparisonStateStorageDependency, id: str)->ApiResult[None]:
  storage.delete(id)
  return ApiResult(data=None, message="The comparison groups has been successfully removed.")


#====================#
# Dashboard
#====================#

def get_dashboard_storage_controller(cache: ProjectCacheDependency):
  return UserDataStorageController[Dashboard](
    path=cache.config.paths.full_path(ProjectPaths.UserData("dashboard")),
    validator=dashboard_validator,
  )

DashboardStorageDependency = Annotated[UserDataStorageController, Depends(get_dashboard_storage_controller)]

@router.get("/dashboard")
def get__all_dashboard(storage: DashboardStorageDependency)->ApiResult[list[UserDataResource[Dashboard]]]:
  return ApiResult(
    data=storage.all(),
    message=None
  )

@router.get("/dashboard/{id}")
def get__dashboard(storage: DashboardStorageDependency, id: str)->ApiResult[UserDataResource[Dashboard]]:
  data = storage.get(id)
  if data is None:
    raise ApiError(f"We were not able to find any dashboard with ID \"{id}\".", HTTPStatus.NOT_FOUND)
  return ApiResult(
    data=data,
    message=None
  )

@router.post("/dashboard")
def post__dashboard(storage: DashboardStorageDependency, body: UserDataSchema[Dashboard])->ApiResult[None]:
  storage.create(body)
  return ApiResult(data=None, message="The dashboard has been successfully created.")

@router.put("/dashboard/{id}")
def put__dashboard(storage: DashboardStorageDependency, id: str, body: UserDataSchema[Dashboard])->ApiResult[None]:
  storage.update(id, body)
  return ApiResult(data=None, message="The dashboard has been successfully updated.")
  
@router.delete("/dashboard/{id}")
def delete__dashboard(storage: DashboardStorageDependency, id: str)->ApiResult[None]:
  storage.delete(id)
  return ApiResult(data=None, message="The dashboard has been successfully removed.")
