from dataclasses import dataclass
import threading
from typing import Any, Callable, Optional

from common.logger import RegisteredLogger
from .requests import TaskRequest
from .responses import TaskLog, TaskResponse, TaskResponseData, TaskStatusEnum

logger = RegisteredLogger().provision("Task")


class IntentionalThreadExit(Exception):
  pass

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
    with self.lock:
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
    raise IntentionalThreadExit()
  
  def error(self, error: Exception):
    logger.error(str(error))
    if self.stop_event.is_set():
      # No need report. Parent process is already aware
      return
    self.stop_event.set()
    raise error


TaskHandlerFn = Callable[[TaskPayload], None]
