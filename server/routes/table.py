from fastapi import APIRouter


router = APIRouter(
  tags=["Table"]
)

@router.post("/{project_id}/table/start")
def post__start_table_processing(project_id: str):
  # Start table preprocessing
  pass

@router.get("/{project_id}/table/status")
def get__table_status(project_id: str):
  # Get table preprocessing status
  pass