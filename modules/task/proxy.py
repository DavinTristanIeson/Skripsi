from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import logging
import multiprocessing
from multiprocessing.synchronize import Event
from typing import Any

from modules.logger.provisioner import ProvisionedLogger
from modules.task.responses import TaskLog, TaskResponse, TaskStatusEnum


class TaskStopException(Exception):
  pass


@dataclass
class TaskManagerProxy:
  id: str
  queue: multiprocessing.Queue
  stop_event: Event
  response: TaskResponse

  def flush(self):
    self.queue.put(self.response.model_dump(), block=True)
  
  @functools.cached_property
  def logger(self):
    return ProvisionedLogger().provision("Task")
  
  def check_stop(self):
    if self.stop_event.is_set():
      raise TaskStopException()

  def log(self, message: str, status: TaskStatusEnum):
    self.logger.info(f"({status}) {message}")
    self.response.logs.append(TaskLog(
      message=message,
      status=status,
    ))
    self.flush()
    self.check_stop()

  def log_success(self, message: str):
    self.log(message, TaskStatusEnum.Success)

  def log_error(self, message: str):
    self.log(message, TaskStatusEnum.Failed)
  
  def log_pending(self, message: str):
    self.log(message, TaskStatusEnum.Pending)
    
  def success(self, data: Any):
    self.logger.info(f"TASK {self.id} SUCCESS: {data}")
    self.response.status = TaskStatusEnum.Success
    self.response.data = data
    self.flush()

  @contextmanager
  def context(self, *, log_file: str):
    ProvisionedLogger().configure(terminal=False, level=logging.DEBUG, file=log_file)
    self.response.status = TaskStatusEnum.Pending
    self.flush()
    try:
      yield
    except TaskStopException as e:
      self.logger.warning(f"Task {self.id} has been cancelled successfully.")
      self.response.logs.append(TaskLog(
        message="Task has been cancelled.",
        status=TaskStatusEnum.Failed,
      ))
      self.response.status = TaskStatusEnum.Failed
    except Exception as e:
      self.logger.exception(e)
      self.response.status = TaskStatusEnum.Failed
      self.response.logs.append(TaskLog(
        message=f"Task failed with the following error: {e}",
        status=TaskStatusEnum.Failed,
      ))
    finally:
      self.flush()

__all__ = [
  "TaskManagerProxy"
]