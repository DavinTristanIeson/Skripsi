import abc
from dataclasses import dataclass
import http
import os
import shutil

from modules.api import ApiError
from modules.exceptions.files import FileNotExistsException
from modules.logger.provisioner import ProvisionedLogger
from modules.storage.atomic import soft_delete
from modules.storage.exceptions import FileSystemCleanupError

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
      raise FileNotExistsException(
        message=f"The file \"{path}\" does not exist. Perhaps the file has not been created yet."
      )
    return path
  
  def allocate_base_path(self)->str:
    dirpath = os.path.dirname(self.base_path)
    if not os.path.exists(dirpath):
      os.makedirs(dirpath, exist_ok=True)
    return self.base_path
  
  def allocate_path(self, path: str)->str:
    full_path = self.full_path(path)
    dirpath = os.path.dirname(full_path)
    os.makedirs(dirpath, exist_ok=True)
    return full_path
  
  def _cleanup(self, directories: list[str], files: list[str], *, soft: bool):
    """``directories`` and ``files`` should be relative to ``base_path``."""
    directories_str = ', '.join(map(lambda dir: f'"{dir}"', directories))
    files_str = ', '.join(map(lambda file: f'"{file}"', files))
    logger.info(f"Cleaning up the following directories: {directories_str}; and files: {files_str}.")
    for rawdir in directories:
      dir = self.full_path(rawdir)
      try:
        if os.path.exists(dir):
          soft_delete(dir, soft=soft)
      except Exception as e:
        raise FileSystemCleanupError(
          path=dir,
          error=e
        )
    for rawfile in files:
      file = self.full_path(rawfile)
      try:
        soft_delete(file, soft=soft)
      except Exception as e:
        raise FileSystemCleanupError(
          path=file,
          error=e
        )
    
    if os.path.exists(self.base_path):
      remaining_files = os.listdir(self.base_path)
      if len(remaining_files) == 0:
        try:
          os.rmdir(self.base_path)
          logger.info(f"Deleted {self.base_path} as there are no remaining files.")
        except Exception as e:
          raise FileSystemCleanupError(
            path=self.base_path,
            error=e
          )
      else:
        logger.warning(f"Skipping the deletion of \"{self.base_path}\" as there are non-deleted files in the folder: {remaining_files}")
  
__all__ = [
  "AbstractPathManager"
]