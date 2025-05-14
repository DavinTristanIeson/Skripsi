from dataclasses import dataclass
import threading
from typing import Optional

from modules.baseclass import Singleton
from modules.logger.provisioner import ProvisionedLogger
from modules.project.exceptions import UnallowedColumnOperationException, UnallowedFileOperationException
from modules.project.paths import ProjectPathManager, ProjectPaths
from filelock import FileLock, BaseFileLock

@dataclass
class GlobalProjectLock:
  file_lock: BaseFileLock
  thread_lock: threading.RLock
  path: Optional[str]
  column: Optional[str]
  timeout: Optional[float]

  @property
  def logger(self):
    return ProvisionedLogger().provision("GlobalProjectLock")

  def __enter__(self):
    try:
      self.file_lock.acquire(timeout=self.timeout)
      self.thread_lock.acquire(timeout=self.timeout if self.timeout is not None else -1)
      self.logger.debug(f"Acquire lock for {self.path or self.column}")
    except TimeoutError as e:
      if self.path is not None:
        raise UnallowedFileOperationException(path=self.path)
      if self.column is not None:
        raise UnallowedColumnOperationException(column=self.column)
      raise e

  def __exit__(self, type, value, traceback):
    self.file_lock.release()
    self.thread_lock.release()
    self.logger.debug(f"Released lock for {self.path or self.column}")

# Lock is applied on a per-project basis.
# Multiple files can be edited at once so we cannot lock on a per-file basis.
# This is mainly used to prevent data races for cache clients.
class ProjectThreadLockManager(metaclass=Singleton):
  __locks: dict[str, threading.RLock]
  def __init__(self):
    self.__locks = {}
  def get(self, project_id: str):
    if project_id not in self.__locks:
      self.__locks[project_id] = threading.RLock()
    return self.__locks[project_id]

class ProjectFileLockManager(metaclass=Singleton):
  # Thread locks work per thread
  __thread_locks: dict[str, threading.RLock]
  def __init__(self):
    self.__thread_locks = {}

  def provision(self, key: str, *, path: Optional[str] = None, column: Optional[str] = None, timeout: Optional[float] = None):
    thread_lock = self.__thread_locks.get(key, None)
    if thread_lock is None:
      thread_lock = threading.RLock()
      self.__thread_locks[key] = thread_lock

    return GlobalProjectLock(
      file_lock=FileLock(key),
      thread_lock=thread_lock,
      column=column,
      path=path,
      timeout=timeout
    )

  def lock_column(self, project_id: str, column: str, *, wait: bool):
    lockpath = ProjectPathManager(project_id=project_id).allocate_path(f"{ProjectPaths.TopicModelingFolder(column)}.lock")
    lock = self.provision(
      key=lockpath,
      column=column,
      timeout=5.0 if not wait else None,
    )
    return lock

  def lock_file(self, project_id: str, path: str, *, wait: bool):
    paths = ProjectPathManager(project_id=project_id)
    file_path = paths.full_path(path)
    lockpath = paths.allocate_path(f"{path}.lock")
    lock = self.provision(
      key=lockpath,
      path=file_path,
      timeout=5.0 if not wait else None,
    )
    return lock
