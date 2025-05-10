from contextlib import _GeneratorContextManager, contextmanager
import os
import shutil
import tempfile
from typing import Literal, overload


from modules.logger.provisioner import ProvisionedLogger

logger = ProvisionedLogger().provision("Atomic File Operations")

@overload
def atomic_write(path: str, mode: Literal['text'])->_GeneratorContextManager[tempfile._TemporaryFileWrapper[str], None, None]:
  ...
@overload
def atomic_write(path: str, mode: Literal['binary'])->_GeneratorContextManager[tempfile._TemporaryFileWrapper[bytes], None, None]:
  ...

@contextmanager
def atomic_write(path: str, *, mode: Literal['text', 'binary']):
  dirpath = os.path.dirname(path)
  if mode == 'text':
    tmp = tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=dirpath, delete=False)
  else:
    tmp = tempfile.NamedTemporaryFile('wb', dir=dirpath, delete=False)
  logger.debug(f"Created temporary file {tmp.name}")
  temp_path = tmp.name
  try:
    with tmp:
      yield tmp

    os.replace(temp_path, path)
    logger.debug(f"Replaced the file at {path} with the file at {temp_path}.")
  finally:
    if os.path.exists(temp_path):
      logger.debug(f"Cleaning up {temp_path}")
      try:
        os.remove(temp_path)
      except OSError as e:
        logger.error(f"Failed to clean up {temp_path} due to the following error: {e}")
        # Ignore error


TRASH_INDICATOR = "trash-"
LOCK_INDICATOR = ".lock"
def soft_delete(path: str, *, soft: bool):
  dirpath, basepath = os.path.split(path)
  soft_delete_path = os.path.join(dirpath, f"{TRASH_INDICATOR}{basepath}")
  try:
    if not os.path.exists(path):
      return
    
    if soft:
      if os.path.isdir(path):
        shutil.move(path, soft_delete_path)
        trash_files = filter(lambda path: path.startswith(TRASH_INDICATOR) or path.endswith(LOCK_INDICATOR), os.listdir(soft_delete_path))
        # No need to store trash or lock files in delete directory
        for trash_file in trash_files:
          os.remove(os.path.join(soft_delete_path, trash_file))
      else:
        os.replace(path, soft_delete_path)
      logger.debug(f"Deleted {path} successfully. A copy of {path} can still be accessed from {soft_delete_path}")
    else:
      if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
      else:
        os.remove(path)
      logger.debug(f"Deleted {path} successfully.")
  except OSError as e:
    logger.error(f"Failed to delete {path} due to the following error: {e}")
    raise e

__all__ = [
  "atomic_write",
  "soft_delete"
]