from contextlib import contextmanager
import threading

from modules.baseclass import Singleton
from modules.project.exceptions import UnallowedColumnOperationException, UnallowedFileOperationException
from modules.project.paths import ProjectPathManager, ProjectPaths
from filelock import BaseFileLock, FileLock

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
  __locks: dict[str, BaseFileLock]
  def __init__(self):
    self.__locks = {}

  def provision(self, lockpath: str):
    lock = self.__locks.get(lockpath, None)
    if lock is not None:
      return lock
    lock = FileLock(lockpath, timeout=5000)
    self.__locks[lockpath] = lock
    return lock
  
  @contextmanager
  def __claim_lock(self, lock: BaseFileLock, *, alt_exception: Exception, wait: bool):
    try:
      lock.acquire(timeout=5000 if not wait else None)
    except TimeoutError:
      raise alt_exception
    
    try:
      yield
      lock.release()
    except Exception as e:
      lock.release()
      raise e

  def lock_column(self, project_id: str, column: str, *, wait: bool):
    lockpath = ProjectPathManager(project_id=project_id).allocate_path(f"{ProjectPaths.Column(column)}.lock")
    lock = self.provision(lockpath)
    return self.__claim_lock(
      lock=lock,
      alt_exception=UnallowedColumnOperationException(column=column),
      wait=wait,
    )

  def lock_file(self, project_id: str, path: str, *, wait: bool):
    paths = ProjectPathManager(project_id=project_id)
    file_path = paths.full_path(path)
    lockpath = paths.allocate_path(f"{path}.lock")
    lock = self.provision(lockpath)
    return self.__claim_lock(
      lock=lock,
      alt_exception=UnallowedFileOperationException(path=file_path),
      wait=wait,
    )