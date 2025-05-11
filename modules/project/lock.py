from contextlib import contextmanager
import threading

from modules.baseclass import Singleton
from modules.project.exceptions import UnallowedColumnOperationException
from modules.project.paths import ProjectPaths

# Lock is applied on a per-project basis.
# Multiple files can be edited at once so we cannot lock on a per-file basis.
class ProjectLockManager(metaclass=Singleton):
  __locks: dict[str, threading.RLock]
  def __init__(self):
    self.__locks = {}
  def get(self, project_id: str):
    if project_id not in self.__locks:
      self.__locks[project_id] = threading.RLock()
    return self.__locks[project_id]
  

class ProjectColumnLockManager(metaclass=Singleton):
  __locks: dict[str, threading.RLock]
  def __init__(self):
    self.__locks = {}
    
  @contextmanager
  def get(self, project_id: str, column: str):
    key = f"{project_id}__{ProjectPaths.Column(column)}"
    if key not in self.__locks:
      self.__locks[key] = threading.RLock()
    lock = self.__locks[key]

    try:
      lock.acquire(timeout=5000)
    except TimeoutError:
      raise UnallowedColumnOperationException(column=column)

    try:
      yield
      lock.release()
    except Exception as e:
      lock.release()
      raise e
