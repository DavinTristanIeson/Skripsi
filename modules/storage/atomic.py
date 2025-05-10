from contextlib import _GeneratorContextManager, contextmanager
import os
import tempfile
from typing import Literal, Optional, overload, ContextManager
from filelock import FileLock


from modules.logger.provisioner import ProvisionedLogger

@overload
def atomic_write(path: str, mode: Literal['text'])->_GeneratorContextManager[tempfile._TemporaryFileWrapper[str], None, None]:
  ...
@overload
def atomic_write(path: str, mode: Literal['binary'])->_GeneratorContextManager[tempfile._TemporaryFileWrapper[bytes], None, None]:
  ...

@contextmanager
def atomic_write(path: str, *, mode: Literal['text', 'binary']):
  logger = ProvisionedLogger().provision("Atomic Write")
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
    logger.debug(f"Replaced the file at {path} with the file at {temp_path}")
  finally:
    if os.path.exists(temp_path):
      logger.debug(f"Cleaning up {temp_path}")
      try:
        os.remove(temp_path)
      except OSError:
        pass

__all__ = [
  "atomic_write",
]