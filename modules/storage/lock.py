import threading

from modules.baseclass import Singleton

# Lock is applied on a per-project basis.
# Multiple files can be edited at once so we cannot lock on a per-file basis.
class FileSystemLockManager(metaclass=Singleton):
  __locks: dict[str, threading.Lock]
  def __init__(self):
    self.__locks = {}
  def lock(self, project_id: str):
    if project_id not in self.__locks:
      self.__locks[project_id] = threading.Lock()
    return self.__locks[project_id]