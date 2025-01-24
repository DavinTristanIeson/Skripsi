from dataclasses import dataclass
import threading
from typing import Any, Callable, Optional

import pydantic
from common.logger import RegisteredLogger
from models.task.requests import TaskRequest
from models.task.responses import TaskResponse, TaskResponseData

class TaskStepTracker(pydantic.BaseModel):
  max_steps: int
  step: int = 0

  @property
  def progress(self):
    return self.step / self.max_steps
  
  def advance(self, offset: int = 1):
    self.step = max(0, min(self.max_steps, self.step + offset))
    return self.progress

logger = RegisteredLogger().provision("Task")

@dataclass
class TaskPayload:
  id: str

  lock: threading.Lock
  results: dict[str, Any]
  stop_event: threading.Event

  request: TaskRequest

  def check_stop(self):
    if self.stop_event.is_set():
      raise Exception("This process has been cancelled.")

  def progress(self, progress: float, message: str):
    with self.lock:
      self.results[self.id] = TaskResponse.Pending(self.id, progress, message)
    self.check_stop()
    
  def success(self, data: TaskResponseData.TypeUnion, message: Optional[str]):
    with self.lock:
      self.results[self.id] = TaskResponse.Success(self.id, data, message)
    self.check_stop()
  
  def error(self, error: Exception):
    if self.stop_event.is_set():
      # No need report. Parent process is already aware
      return
    with self.lock:
      self.results[self.id] = TaskResponse.Error(self.id, str(error))
    self.stop_event.set()


TaskHandlerFn = Callable[[TaskPayload], None]
