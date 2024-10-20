from dataclasses import dataclass
import functools
import queue
import threading
from typing import Callable, Optional

import pydantic
from common.ipc.requests import IPCRequest
from common.ipc.responses import IPCResponse, IPCResponseData, IPCResponseStatus
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
  pipe: queue.Queue
  stop_event: threading.Event

  request: IPCRequest

  def progress(self, progress: float, message: str):
    def send_progress():
      response = IPCResponse.Pending(self.id, progress, message)
      self.pipe.put(response)
    thread = threading.Thread(target=send_progress)
    thread.start()
    return thread
    
  def success(self, data: IPCResponseData.TypeUnion, message: Optional[str]):
    self.pipe.put(IPCResponse.Success(self.id, data, message))
  
  def error(self, error: Exception):
    if self.stop_event.is_set():
      # No need report. Parent process is already aware
      return
    self.pipe.put(IPCResponse.Error(self.id, str(error)))
    self.stop_event.set()
  
  def check_stop(self):
    if self.stop_event.is_set():
      raise Exception("This process has been cancelled.")


IPCTaskHandlerFn = Callable[[IPCTask], None]

