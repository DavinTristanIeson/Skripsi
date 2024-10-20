import http
import os
from fastapi import APIRouter
from pydantic import RootModel
from common.models.api import ApiResult
from wordsmith.data import paths
from wordsmith.data import source

class DataSourceWrapper(RootModel):
  root: source.DataSource

router = APIRouter(
  prefix="/projects/check",
  tags=['check']
)

@router.post(
  "/project-name/{project_name}", 
  status_code=http.HTTPStatus.OK, 
)
async def check_project(project_name: str):
  folder_name = paths.DATA_DIRECTORY
  folder_path = os.path.join(os.getcwd(), folder_name, project_name)
  if os.path.isdir(folder_path):
    available = False
    message = f"The project name '{project_name}' is already taken. Please choose a different name."
  else:
    available = True
    message = f"The project name '{project_name}' is available. You're good to go!"
  
  return ApiResult(data={"available": available}, message=f"{message}")

@router.post(
  "/dataset-path"
)
async def check_dataset(body: DataSourceWrapper):
  return {"data": body.root.load()}
    