import abc
from dataclasses import dataclass
import http
import os
import shutil

from modules.api import ApiError
from modules.logger.provisioner import ProvisionedLogger

logger = ProvisionedLogger().provision("AbstractPathManager")
class AbstractPathManager(abc.ABC):
  @property
  @abc.abstractmethod
  def base_path(self)->str:
    pass

  def full_path(self, path: str)->str:
    project_dir = self.base_path
    fullpath:str = os.path.join(os.getcwd(), project_dir, path)
    return fullpath
  
  def assert_path(self, path: str)->str:
    path = self.full_path(path)
    if not os.path.exists(path):
      raise ApiError(f"{path} does not exist. Perhaps the file has not been created yet.", 404)
    return path
  
  def allocate_path(self, path: str)->str:
    full_path = self.full_path(path)
    dirpath = os.path.dirname(full_path)
    os.makedirs(dirpath, exist_ok=True)
    return full_path
  
  def _cleanup(self, directories: list[str], files: list[str]):
    """``directories`` and ``files`` should be relative to ``base_path``."""
    directories_str = ', '.join(map(lambda dir: f'"{dir}"', directories))
    files_str = ', '.join(map(lambda file: f'"{file}"', files))
    logger.info(f"Cleaning up the following directories: {directories_str}; and files: {files_str}.")
    try:
      for rawdir in directories:
        dir = self.full_path(rawdir)
        if os.path.exists(dir):
          shutil.rmtree(dir)
          logger.debug(f"Deleted directory: \"{dir}\".")
      for rawfile in files:
        file = self.full_path(rawfile)
        if os.path.exists(file):
          os.remove(file)
          logger.debug(f"Deleted file: \"{file}\".")
    except Exception as e:
      logger.error(f"An error has occurred while deleting directories and/or files from \"{self.base_path}\". Error => {e}")
      raise ApiError(f"An unexpected error has occurred while cleaning up the \"{self.base_path}\" folder: {e}", http.HTTPStatus.INTERNAL_SERVER_ERROR)
    
    if os.path.exists(self.base_path):
      remaining_files = os.listdir(self.base_path)
      if len(remaining_files) == 0:
        try:
          os.rmdir(self.base_path)
          logger.info(f"Deleted {self.base_path} as there are no remaining files.")
        except ApiError as e:
          logger.error(f"An error has occurred while deleting \"{self.base_path}\". Error => {e}")
          raise ApiError(f"An unexpected error has occurred while cleaning up the \"{self.base_path}\" folder: {e}", http.HTTPStatus.INTERNAL_SERVER_ERROR)
      else:
        logger.warning(f"Skipping the deletion of \"{self.base_path}\" as there are non-managed files in the folder: {remaining_files}")
  
__all__ = [
  "AbstractPathManager"
]