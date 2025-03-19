from dataclasses import dataclass
import http
import threading
from typing import Callable

from modules.baseclass import Singleton
from modules.logger import ProvisionedLogger
from modules.api import ApiError
from .requests import TaskRequest
from .responses import TaskLog, TaskResponse, TaskResponseData, TaskStatusEnum

logger = ProvisionedLogger().provision("Task")

@dataclass
class TaskStorageProxy:
  id: str
  lock: threading.Lock
  results: dict[str, TaskResponse]

  @property
  def task(self)->"TaskResponse":
    if self.id not in self.results:
      raise ApiError(
        f"The task \"{self.id}\" has not been created yet. This should be a developer oversight. Try re-executing the procedure again.",
        http.HTTPStatus.INTERNAL_SERVER_ERROR
      )
    return self.results[self.id]

  def log(self, message: str, status: TaskStatusEnum):
    logger.info(f"({status}) {message}")
    with self.lock:
      self.task.logs.append(TaskLog(
        message=message,
        status=status,
      ))

  def log_success(self, message: str):
    self.log(message, TaskStatusEnum.Success)

  def log_error(self, message: str):
    self.log(message, TaskStatusEnum.Failed)
  
  def log_pending(self, message: str):
    self.log(message, TaskStatusEnum.Pending)
    
  def success(self, data: TaskResponseData.TypeUnion):
    logger.info(data)
    with self.lock:
      task = self.task
      task.status = TaskStatusEnum.Success
      task.data = data

TaskHandlerFn = Callable[[TaskStorageProxy], None]

class TaskStorage(metaclass=Singleton):
  results: dict[str, TaskResponse]
  lock: threading.Lock
  def __init__(self) -> None:
    self.results = {}
    self.lock = threading.Lock()
  def proxy(self, result_id: str):
    return TaskStorageProxy(
      id=result_id,
      lock=self.lock,
      results=self.results,
    )


__all__ = [
  "TaskStorageProxy"
]