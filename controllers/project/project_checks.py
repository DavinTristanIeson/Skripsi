import http
import os
from models.project import CheckProjectIdSchema
from modules.api.wrapper import ApiError, ApiResult
from modules.config.paths import ProjectPathManager


def check_if_project_exists(body: CheckProjectIdSchema):
  paths = ProjectPathManager(
    project_id=body.project_id
  )
  # Make sure data directory exists
  paths.allocate_base_path()

  if os.path.isdir(paths.project_path):
    message = f"The project name \"{body.project_id}\" is already taken. Please choose a different name."
    raise ApiError(message, http.HTTPStatus.BAD_REQUEST)
  else:
    message = f"The project name \"{body.project_id}\" is available. You're good to go!"
    return ApiResult(data=None, message=f"{message}")