import http
import os
from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
from common.models.api import ApiResult
from server.models.project import CheckDatasetResource, CheckDatasetSchema, CheckProjectIdSchema, DatasetInferredColumnResource
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
  folder_name = paths.DATA_DIRECTORY
  folders = []

  if os.path.isdir(folder_name):
    folders = [name for name in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, name))]

  return ApiResult(
    data=folders,
    message="Success"
  )

@router.get('/{project_id}')
async def get__project(project_id: str):
  try:
    config = Config.from_project(project_id)

    return ApiResult(
      data=config,
      message="Success"
    )
  except FileNotFoundError:
    raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))