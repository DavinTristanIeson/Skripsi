from fastapi import APIRouter
from fastapi.responses import FileResponse


router = APIRouter(
  tags=['Application']
)

VIEW_FOLDER = "views"


@router.get('/projects/{project_id}')
def get__projects_page():
  return FileResponse(path=f"{VIEW_FOLDER}/projects/[id].html")

@router.get('/projects/{project_id}/config')
def get__config_page():
  return FileResponse(path=f"{VIEW_FOLDER}/projects/[id]/config.html")

@router.get('/projects/{project_id}/topics')
def get__topics_page():
  return FileResponse(path=f"{VIEW_FOLDER}/projects/[id]/topics.html")

@router.get('/projects/{project_id}/topics/evaluation')
def get__topic_evaluation_page():
  return FileResponse(path=f"{VIEW_FOLDER}/projects/[id]/topics/evaluation.html")

@router.get('/projects/{project_id}/topics/experiment')
def get__topic_model_experiments_page():
  return FileResponse(path=f"{VIEW_FOLDER}/projects/[id]/topics/experiment.html")

@router.get('/projects/{project_id}/topics/refine')
def get__refine_topics_page():
  return FileResponse(path=f"{VIEW_FOLDER}/projects/[id]/topics/refine.html")

@router.get('/projects/{project_id}/table')
def get__table_page():
  return FileResponse(path=f"{VIEW_FOLDER}/projects/[id]/table.html")

@router.get('/projects/{project_id}/comparison')
def get__comparison_page():
  return FileResponse(path=f"{VIEW_FOLDER}/projects/[id]/comparison.html")

@router.get('/projects/{project_id}/create')
def get__create_page():
  return FileResponse(path=f"{VIEW_FOLDER}/projects/[id]/create.html")