from dataclasses import dataclass
import http
import threading
from typing import Callable

from modules.logger import ProvisionedLogger
from modules.api import ApiError
from .requests import TaskRequest
from .responses import TaskLog, TaskResponse, TaskResponseData, TaskStatusEnum

logger = ProvisionedLogger().provision("Task")

@dataclass
class TaskPayload:
  id: str

  lock: threading.Lock
  results: dict[str, TaskResponse]
  stop_event: threading.Event

  request: TaskRequest

  def check_stop(self):
    if self.stop_event.is_set():
      raise Exception("This process has been cancelled.")
    
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
    self.check_stop()

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
    self.check_stop()

TaskHandlerFn = Callable[[TaskPayload], None]

__all__ = [
  "TaskPayload"
]