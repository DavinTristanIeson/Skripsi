from contextlib import _GeneratorContextManager, contextmanager
import os
import shutil
import tempfile
from typing import Literal, overload


from modules.logger.provisioner import ProvisionedLogger

logger = ProvisionedLogger().provision("Atomic File Operations")

TRASH_INDICATOR = ".trash"
LOCK_INDICATOR = ".lock"
def is_unimportant_file_path(path: str):
  return path.endswith(TRASH_INDICATOR) or path.endswith(LOCK_INDICATOR)

def replace_or_rename(src_path: str, dest_path: str):
  try:
    os.replace(src_path, dest_path)
  except OSError as e:
    logger.debug(f"Failed to use os.replace due to {e}. Falling back to os.rename.")
    os.rename(src_path, dest_path)

def soft_delete_directory(src_path: str, dest_path: str):
  if os.path.exists(dest_path):
    shutil.rmtree(dest_path)
  replace_or_rename(src_path, dest_path)
  trash_files = filter(
    is_unimportant_file_path,
    os.listdir(dest_path)
  )
  # No need to store trash or lock files in delete directory
  for trash_file in trash_files:
    os.remove(os.path.join(dest_path, trash_file))

def soft_delete(path: str, *, soft: bool):
  soft_delete_path = f"{path}{TRASH_INDICATOR}"
  try:
    if not os.path.exists(path):
      return
    
    if soft:
      if os.path.isdir(path):
        # Clean up trash
        soft_delete_directory(path, soft_delete_path)
      else:
        replace_or_rename(path, soft_delete_path)
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
  
@overload
def atomic_write(path: str, mode: Literal['text'])->_GeneratorContextManager[tempfile._TemporaryFileWrapper, None, None]:
  ...
@overload
def atomic_write(path: str, mode: Literal['binary'])->_GeneratorContextManager[tempfile._TemporaryFileWrapper, None, None]:
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

    trash_path = f"{path}{TRASH_INDICATOR}"
    try:
      replace_or_rename(path, trash_path)
      logger.debug(f"Created a trash backup of {path} in {trash_path}")
    except OSError as e:
      logger.error(f"Failed to create a trash backup of {path} in {trash_path} due to {e}")
      
    replace_or_rename(temp_path, path)
    logger.debug(f"Replaced the file at {path} with the file at {temp_path}.")
  finally:
    if os.path.exists(temp_path):
      try:
        os.remove(temp_path)
      except OSError as e:
        logger.error(f"Failed to clean up {temp_path} due to the following error: {e}")
        # Ignore error


__all__ = [
  "atomic_write",
  "soft_delete"
]