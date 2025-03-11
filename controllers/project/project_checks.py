import http
import os
from models.project import CheckProjectIdSchema
from modules.api.wrapper import ApiError, ApiResult
from modules.logger.provisioner import ProvisionedLogger
from modules.project.paths import DATA_DIRECTORY, ProjectPathManager

logger = ProvisionedLogger().provision("Project Controller")
def _assert_valid_project_id(project_id: str):
  paths = ProjectPathManager(project_id=project_id)
  if os.path.isdir(paths.project_path):
    message = f"The project name \"{project_id}\" is already taken. Please choose a different name."
    raise ApiError(message, http.HTTPStatus.BAD_REQUEST)
  try:
    os.makedirs(paths.project_path, exist_ok=False)
    os.rmdir(paths.project_path)
  except OSError as e:
    logger.error(f"check_if_project_exists failed with error: {e}")
    message = f"The project name \"{project_id}\" is not a valid folder name. Please choose a different name."
    raise ApiError(message, http.HTTPStatus.BAD_REQUEST)

def check_if_project_exists(body: CheckProjectIdSchema):
  paths = ProjectPathManager(
    project_id=body.project_id
  )
  # Make sure data directory exists
  paths.allocate_base_path()
  _assert_valid_project_id(body.project_id)
  message = f"The project name \"{body.project_id}\" is available. You're good to go!"
  return ApiResult(data=None, message=f"{message}")

__all__ = [
  "check_if_project_exists",
]