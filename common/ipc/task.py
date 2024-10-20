from dataclasses import dataclass
import threading
from typing import Callable, Optional

import pydantic
from common.ipc.client import IntraProcessCommunicator
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
  comm: IntraProcessCommunicator
  request: IPCRequest

  def progress(self, progress: float, message: str):
    def send_progress():
      response = IPCResponse(
        data=IPCResponseData.Empty(),
        error=None,
        message=message,
        progress=progress,
        status=IPCResponseStatus.Pending,
        id=self.id,
      )
      logger.debug(f"Sending {response.model_dump_json()} from IPC task handler to parent process.")
      self.comm.pipe.send(response)
    thread = threading.Thread(target=send_progress)
    thread.start()
    return thread
    
  def success(self, data: IPCResponseData.TypeUnion, message: Optional[str]):
    self.comm.pipe.send(IPCResponse(
      data=data,
      error=None,
      message=message,
      progress=1,
      status=IPCResponseStatus.Success,
      id=self.id,
    ))
  
  def error(self, error: Exception):
    if self.comm.stop_event.is_set():
      # No need report. Parent process is already aware
      return
    self.comm.pipe.send(IPCResponse(
      data=IPCResponseData.Empty(),
      error=str(error),
      message=None,
      progress=1,
      status=IPCResponseStatus.Failed,
      id=self.id,
    ))
    self.comm.stop_event.set()
  
  def check_stop(self):
    if self.comm.stop_event.is_set():
      raise Exception("This process has been cancelled.")


IPCTaskHandlerFn = Callable[[IPCTask], None]

def ipc_task_handler(fn: IPCTaskHandlerFn):
  def inner(comm: IPCTask):
    try:
      fn(comm)
    except Exception as e:
      comm.error(e)
  return inner

