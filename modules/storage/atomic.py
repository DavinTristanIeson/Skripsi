from contextlib import _GeneratorContextManager, contextmanager
import os
import tempfile
from typing import Literal, overload


from modules.logger.provisioner import ProvisionedLogger

logger = ProvisionedLogger().provision("Atomic File Operations")

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

    if os.path.exists(path):
      os.replace(temp_path, path)
    else:
      os.rename(temp_path, path)
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
]