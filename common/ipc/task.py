from dataclasses import dataclass
import queue
import threading
from typing import Callable, Optional

import pydantic
from common.ipc.requests import IPCRequest
from common.ipc.responses import IPCResponse, IPCResponseData
from common.logger import RegisteredLogger

class TaskStepTracker(pydantic.BaseModel):
  max_steps: int
  step: int = 0

  @property
  def progress(self):
    return self.step / self.max_steps
  
  def advance(self, offset: int = 1):
    self.step = max(0, min(self.max_steps, self.step + offset))
    return self.progress

logger = RegisteredLogger().provision("IPC")

@dataclass
class IPCTask:
  id: str

  lock: threading.Lock
  results: dict[str, IPCResponse]
  stop_event: threading.Event

  request: IPCRequest

  def progress(self, progress: float, message: str):
    with self.lock:
      self.results[self.id] = IPCResponse.Pending(self.id, progress, message)
    
  def success(self, data: IPCResponseData.TypeUnion, message: Optional[str]):
    with self.lock:
      self.results[self.id] = IPCResponse.Success(self.id, data, message)
  
  def error(self, error: Exception):
    if self.stop_event.is_set():
      # No need report. Parent process is already aware
      return
    with self.lock:
      self.results[self.id] = IPCResponse.Error(self.id, str(error))
    self.stop_event.set()
  
  def check_stop(self):
    if self.stop_event.is_set():
      raise Exception("This process has been cancelled.")


IPCTaskHandlerFn = Callable[[IPCTask], None]

