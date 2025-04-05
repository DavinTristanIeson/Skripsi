from http import HTTPStatus
from typing import Annotated, Any
from fastapi import APIRouter, Depends

from controllers.project.dependency import ProjectCacheDependency
from controllers.userdata import (
  JSONStorageController
)
from models.userdata import (
  ComparisonState, UserDataResource, UserDataSchema
)
from modules.api.wrapper import ApiError, ApiResult
from modules.dashboard import Dashboard
from modules.project.paths import ProjectPaths
from modules.table.filter_variants import TableFilter

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
  return JSONStorageController[TableFilter](
    path=cache.config.paths.full_path(ProjectPaths.UserData("filters")),
    validator=filter_validator,
  )

FiltersStorageDependency = Annotated[JSONStorageController, Depends(get_filters_storage_controller)]

@router.get("/filters")
def get__all_filters(storage: FiltersStorageDependency)->ApiResult[list[UserDataResource[TableFilter]]]:
  return ApiResult(
    data=storage.all(),
    message=None
  )

@router.get("/filters/{name}")
def get__filters(storage: FiltersStorageDependency, name: str)->ApiResult[UserDataResource[TableFilter]]:
  data = storage.get(name)
  if data is None:
    raise ApiError(f"We were not able to find any filter with name \"{name}\".", HTTPStatus.NOT_FOUND)
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
  return JSONStorageController[ComparisonState](
    path=cache.config.paths.full_path(ProjectPaths.UserData("comparison_state")),
    validator=comparison_state_validator,
  )

ComparisonStateStorageDependency = Annotated[JSONStorageController, Depends(get_comparison_state_storage_controller)]

@router.get("/comparison-state")
def get__all_comparison_state(storage: ComparisonStateStorageDependency)->ApiResult[list[UserDataResource[ComparisonState]]]:
  return ApiResult(
    data=storage.all(),
    message=None
  )

@router.get("/comparison-state/{name}")
def get__comparison_state(storage: ComparisonStateStorageDependency, name: str)->ApiResult[UserDataResource[ComparisonState]]:
  data = storage.get(name)
  if data is None:
    raise ApiError(f"We were not able to find any comparison groups with name \"{name}\".", HTTPStatus.NOT_FOUND)
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
# TableDashboard
#====================#

def get_table_dashboard_storage_controller(cache: ProjectCacheDependency):
  return JSONStorageController[Dashboard](
    path=cache.config.paths.full_path(ProjectPaths.UserData("table_dashboard")),
    validator=dashboard_validator,
  )

TableDashboardStorageDependency = Annotated[JSONStorageController, Depends(get_table_dashboard_storage_controller)]

@router.get("/dashboard/table")
def get__all_table_dashboard(storage: TableDashboardStorageDependency)->ApiResult[list[UserDataResource[Dashboard]]]:
  return ApiResult(
    data=storage.all(),
    message=None
  )

@router.get("/dashboard/table/{name}")
def get__table_dashboard(storage: TableDashboardStorageDependency, name: str)->ApiResult[UserDataResource[Dashboard]]:
  data = storage.get(name)
  if data is None:
    raise ApiError(f"We were not able to find any table dashboard with name \"{name}\".", HTTPStatus.NOT_FOUND)
  return ApiResult(
    data=data,
    message=None
  )

@router.post("/dashboard/table")
def post__table_dashboard(storage: TableDashboardStorageDependency, body: UserDataSchema[Dashboard])->ApiResult[None]:
  storage.create(body)
  return ApiResult(data=None, message="The table dashboard has been successfully created.")

@router.put("/dashboard/table/{id}")
def put__table_dashboard(storage: TableDashboardStorageDependency, id: str, body: UserDataSchema[Dashboard])->ApiResult[None]:
  storage.update(id, body)
  return ApiResult(data=None, message="The table dashboard has been successfully updated.")
  
@router.delete("/dashboard/table/{id}")
def delete__table_dashboard(storage: TableDashboardStorageDependency, id: str)->ApiResult[None]:
  storage.delete(id)
  return ApiResult(data=None, message="The table dashboard has been successfully removed.")


#====================#
# ComparisonDashboard
#====================#

def get_comparison_dashboard_storage_controller(cache: ProjectCacheDependency):
  return JSONStorageController[Dashboard](
    path=cache.config.paths.full_path(ProjectPaths.UserData("comparison_dashboard")),
    validator=dashboard_validator,
  )

ComparisonDashboardStorageDependency = Annotated[JSONStorageController, Depends(get_comparison_dashboard_storage_controller)]

@router.get("/dashboard/comparison")
def get__all_comparison_dashboard(storage: ComparisonDashboardStorageDependency)->ApiResult[list[UserDataResource[Dashboard]]]:
  return ApiResult(
    data=storage.all(),
    message=None
  )

@router.get("/dashboard/comparison/{name}")
def get__comparison_dashboard(storage: ComparisonDashboardStorageDependency, name: str)->ApiResult[UserDataResource[Dashboard]]:
  data = storage.get(name)
  if data is None:
    raise ApiError(f"We were not able to find any comparison dashboard with name \"{name}\".", HTTPStatus.NOT_FOUND)
  return ApiResult(
    data=data,
    message=None
  )

@router.post("/dashboard/comparison")
def post__comparison_dashboard(storage: ComparisonDashboardStorageDependency, body: UserDataSchema[Dashboard])->ApiResult[None]:
  storage.create(body)
  return ApiResult(data=None, message="The comparison dashboard has been successfully created.")

@router.put("/dashboard/comparison/{id}")
def put__comparison_dashboard(storage: ComparisonDashboardStorageDependency, id: str, body: UserDataSchema[Dashboard])->ApiResult[None]:
  storage.update(id, body)
  return ApiResult(data=None, message="The comparison dashboard has been successfully updated.")
  
@router.delete("/dashboard/comparison/{id}")
def delete__comparison_dashboard(storage: ComparisonDashboardStorageDependency, id: str)->ApiResult[None]:
  storage.delete(id)
  return ApiResult(data=None, message="The comparison dashboard has been successfully removed.")


#====================#
# CorrelationDashboard
#====================#

def get_correlation_dashboard_storage_controller(cache: ProjectCacheDependency):
  return JSONStorageController[Dashboard](
    path=cache.config.paths.full_path(ProjectPaths.UserData("correlation_dashboard")),
    validator=dashboard_validator,
  )

CorrelationDashboardStorageDependency = Annotated[JSONStorageController, Depends(get_correlation_dashboard_storage_controller)]

@router.get("/dashboard/correlation")
def get__all_correlation_dashboard(storage: CorrelationDashboardStorageDependency)->ApiResult[list[UserDataResource[Dashboard]]]:
  return ApiResult(
    data=storage.all(),
    message=None
  )

@router.get("/dashboard/correlation/{name}")
def get__correlation_dashboard(storage: CorrelationDashboardStorageDependency, name: str)->ApiResult[UserDataResource[Dashboard]]:
  data = storage.get(name)
  if data is None:
    raise ApiError(f"We were not able to find any topic correlation dashboard with name \"{name}\".", HTTPStatus.NOT_FOUND)
  return ApiResult(
    data=data,
    message=None
  )

@router.post("/dashboard/correlation")
def post__correlation_dashboard(storage: CorrelationDashboardStorageDependency, body: UserDataSchema[Dashboard])->ApiResult[None]:
  storage.create(body)
  return ApiResult(data=None, message="The topic correlation dashboard has been successfully created.")

@router.put("/dashboard/correlation/{id}")
def put__correlation_dashboard(storage: CorrelationDashboardStorageDependency, id: str, body: UserDataSchema[Dashboard])->ApiResult[None]:
  storage.update(id, body)
  return ApiResult(data=None, message="The topic correlation dashboard has been successfully updated.")
  
@router.delete("/dashboard/correlation/{id}")
def delete__correlation_dashboard(storage: CorrelationDashboardStorageDependency, id: str)->ApiResult[None]:
  storage.delete(id)
  return ApiResult(data=None, message="The topic correlation dashboard has been successfully removed.")
