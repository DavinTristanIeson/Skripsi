import http
import shutil
import os
from fastapi import APIRouter, HTTPException, Query
import numpy as np
import pandas as pd
from common.models.api import ApiResult, ApiError
from common.ipc.taskqueue import IPCTaskLocker
from common.ipc.requests import IPCRequestData
from common.ipc.client import SERVER2TOPIC_IPC_CHANNEL, TOPIC2SERVER_IPC_CHANNEL
from server.models.project import CheckDatasetResource, CheckDatasetSchema, CheckProjectIdSchema, DatasetInferredColumnResource, ProjectLiteResource, ProjectResource
from wordsmith.data import paths
from wordsmith.data.schema import SchemaColumnTypeEnum
from wordsmith.data.config import Config
 
router = APIRouter(
  tags=['Projects']
)

@router.post(
  "/check-project-id", 
  status_code=http.HTTPStatus.OK, 
)
async def check_project(body: CheckProjectIdSchema):
  folder_name = paths.DATA_DIRECTORY
  folder_path = os.path.join(os.getcwd(), folder_name, body.project_id)
  if os.path.isdir(folder_path):
    available = False
    message = f"The project name \"{body.project_id}\" is already taken. Please choose a different name."
  else:
    available = True
    message = f"The project name \"{body.project_id}\" is available. You're good to go!"
  
  return ApiResult(data={"available": available}, message=f"{message}")

@router.post("/check-dataset")
async def check_dataset(body: CheckDatasetSchema):
  df = body.root.load()
  columns: list[DatasetInferredColumnResource] = []
  for column in df.columns:
    dtype = df[column].dtype
    coltype: SchemaColumnTypeEnum
    if dtype == np.float_ or dtype == np.int_:
      coltype = SchemaColumnTypeEnum.Continuous
    else:
      uniquescnt = len(df[column].unique())
      if uniquescnt < 0.2 * len(df[column]):
        coltype = SchemaColumnTypeEnum.Categorical
      else:
        has_long_text = df[column].str.len().mean() >= 20
        if has_long_text:
          coltype = SchemaColumnTypeEnum.Textual
        else:
          coltype = SchemaColumnTypeEnum.Unique

    columns.append(DatasetInferredColumnResource(
      name=column,
      type=coltype,
    ))

  return ApiResult(
    data=CheckDatasetResource(columns=columns),
    message=f"We have inferred the columns from the dataset at {body.root.path}. Next, please configure how you would like to process the individual columns."
  )
    
@router.get('/')
async def get__projects():
  folder_name = os.path.join(os.getcwd(), paths.DATA_DIRECTORY)
  projects: list[ProjectLiteResource] = []

  if os.path.isdir(folder_name):
    folders = [
      name for name in os.listdir(folder_name)
      if os.path.isdir(os.path.join(folder_name, name))
      # don't include hidden folders
      and not folder_name.startswith('.')
    ]
    projects = list(map(lambda folder: ProjectLiteResource(id=folder, path=os.path.join(folder_name, folder)), folders))

  return ApiResult(
    data=projects,
    message=None
  )

@router.get('/{project_id}')
async def get__project(project_id: str):
  config = Config.from_project(project_id)
  return ApiResult(
    data=ProjectResource(
      id=project_id,
      config=config,
    ),
    message=None
  )

@router.post('/')
async def create__project(config: Config):
  folder_path = os.path.join(os.getcwd(), paths.DATA_DIRECTORY, config.project_id)
  os.makedirs(paths.DATA_DIRECTORY, exist_ok=True)

  if os.path.isdir(folder_path):
    raise ApiError(f"Project \"{config.project_id}\" already exists. Try another name.", http.HTTPStatus.UNPROCESSABLE_ENTITY)

  os.makedirs(folder_path)
  config.save_to_json(folder_path=folder_path)
  return ApiResult(
    data=ProjectResource(
      id=config.project_id,
      config=config
    ),
    message=f"Your new project \"{config.project_id}\", has been successfully created."
  )

@router.put('/{project_id}')
async def update__project(project_id: str, config: Config):
  folder_path = os.path.join(os.getcwd(), paths.DATA_DIRECTORY, project_id)
  if not os.path.isdir(folder_path):
    raise ApiError(f"Project '{project_id}' not found!", 404)

  if config.project_id != project_id:
    old_folder_path = folder_path
    folder_path = os.path.join(os.getcwd(), paths.DATA_DIRECTORY, config.project_id)

    if os.path.isdir(folder_path):
      raise ApiError(f"Project '{config.project_id}' already exists. Try another name.", http.HTTPStatus.UNPROCESSABLE_ENTITY)
    
    os.rename(old_folder_path, folder_path)

  config.save_to_json(folder_path=folder_path)

  return ApiResult(
    data=ProjectResource(
      id=project_id,
      config=config,
    ),
    message=f"Project \"{project_id}\" has been successfully updated."
  )

@router.delete('/{project_id}')
async def delete__project(project_id: str):
  folder_path = os.path.join(os.getcwd(), paths.DATA_DIRECTORY, project_id)

  if not os.path.isdir(folder_path):
    raise ApiError(f"We cannot find any projects with ID: \"{project_id}\", perhaps it had been manually deleted by a user?", 404)

  shutil.rmtree(folder_path)
  return ApiResult(
    data=None,
    message=f"Project \"{project_id}\" has been successfully deleted."
  )

@router.get('/{project_id}/association')
async def get__association(project_id: str, column1: str = Query(...), column2: str = Query(...)):
  folder_path = os.path.join(os.getcwd(), paths.DATA_DIRECTORY, project_id)

  if not os.path.isdir(folder_path):
    raise ApiError(f"We cannot find any projects with ID: \"{project_id}\"", 404)
  
  config = Config.from_project(project_id=project_id)

  column1 = config.dfschema.assert_exists(column1)
  column2 = config.dfschema.assert_exists(column2)
  if (column1.type is not SchemaColumnTypeEnum.Textual):
    raise ApiError("Please fill column1 with textual type of column", 400)
  if (column2.type is SchemaColumnTypeEnum.Unique):
    raise ApiError("Please fill column2 with non-unique type of column", 400)

  workspace = config.paths.load_workspace()
  if column1.name not in workspace.columns:
    raise ApiError(f"Column {column1.name} could not be found in the current workspace. Please consider updating to a new workspace!", 404)
  if column2.name not in workspace.columns:
    raise ApiError(f"Column {column2.name} could not be found in the current workspace. Please consider updating to a new workspace!", 404)
  
  task_id = f"association-plot-{project_id}"

  locker = IPCTaskLocker()

  locker.initialize(
    channel=SERVER2TOPIC_IPC_CHANNEL,
    backchannel=TOPIC2SERVER_IPC_CHANNEL
  )

  # kurang paham ini perlu atau tidak
  # if result:=locker.result(task_id):
  #   return ApiResult(data=result, message=result.message)

  locker.request(IPCRequestData.AssociationPlot(
    id=task_id,
    project_id=config.project_id,
    col1=column1.name,
    col2=column2.name,
  ))

  return ApiResult(
    data=None,
    message="..."
  )