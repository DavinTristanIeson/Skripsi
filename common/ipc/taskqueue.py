import concurrent.futures
import queue
import threading
from typing import Any, Optional, cast

from pydantic import ValidationError

from common.ipc.client import IPCChannel, IPCClient, IPCListener
from common.ipc.operations import IPCOperationRequest, IPCOperationRequestData, IPCOperationRequestType, IPCOperationRequestWrapper, IPCOperationResponse, IPCOperationResponseData, IPCOperationResponseWrapper
from common.ipc.requests import IPCRequest, IPCRequestType, IPCRequestWrapper
from common.ipc.responses import IPCResponse, IPCResponseStatus
from common.ipc.task import IPCTask, IPCTaskHandlerFn
from common.logger import RegisteredLogger, TimeLogger
from common.models.metaclass import Singleton

logger = RegisteredLogger().provision("IPC")

class IPCTaskServer(metaclass=Singleton):
  pool: concurrent.futures.ThreadPoolExecutor
  handlers: dict[IPCRequestType, IPCTaskHandlerFn]
  results: dict[str, IPCResponse]
  listener: IPCListener
  lock: threading.Lock
  channel: IPCChannel
  ongoing_tasks: dict[str, threading.Event]

  def __init__(self) -> None:
    super().__init__()
    self.ongoing_tasks = {}
    self.lock = threading.Lock()

  def handle_task(self, handler: IPCTaskHandlerFn, message: IPCRequest):
    stop_event = threading.Event()
    with self.lock:
      self.ongoing_tasks[message.id] = stop_event
    with TimeLogger(logger, f"Handling task {message.id} with payload: {message.model_dump_json()}"):
      try:
        handler(
          IPCTask(
            id=message.id,
            lock=self.lock,
            results=self.results,
            stop_event=stop_event,
            request=message
          ),
        )
      except Exception as e:
        logger.error(f"An error has occurred during the execution of task {message.id}. Error: {e}")
        with self.lock:
          self.results[message.id] = IPCResponse.Error(message.id, str(e))
        return
      
    if message.id in self.ongoing_tasks:
      with self.lock:
        self.ongoing_tasks.pop(message.id)
      logger.debug(f"Cleaned up task for {message.id}.")
    else:
      logger.warning(f"The ongoing task for task {message.id} had been removed unexpectedly.")

  def cancel_task(self, id: str):
    with self.lock:
      if id not in self.ongoing_tasks:
        return  
      self.ongoing_tasks[id].set()
      self.ongoing_tasks.pop(id)
    
  def on_received_request(self, request: IPCRequest):
    handler = self.handlers.get(request.type, None)
    if handler is None:
      logger.error(f"No handler for message of type {request.type} has been registered!")
      return
    
    with self.lock:
      has_task_id = request.id in self.ongoing_tasks
    if has_task_id:
      logger.warning(f"Canceling ongoing task for {request.id} in lieu of the new task: {request.model_dump_json()}")
      self.cancel_task(request.id)

    with self.lock:
      response = IPCResponse.Idle(request.id)
      self.results[request.id] = response

    self.pool.submit(self.handle_task, handler, request)
    return response

  def on_received_operation(self, request: IPCOperationRequest)->IPCOperationResponse:
    if request.type == IPCOperationRequestType.GetResult:
      return IPCOperationResponseData.Result(data=self.results.get(request.id, None))
    
    if request.type == IPCOperationRequestType.CancelTask:
      self.cancel_task(request.id)
      return IPCOperationResponseData.Empty()
    
    if request.type == IPCOperationRequestType.SanityCheck:
      sanity = IPCResponse.Idle(request.id)
      with self.lock:
        self.results[request.id] = sanity
      return IPCOperationResponseData.Result(data=self.results[request.id])
    
    if request.type == IPCOperationRequestType.SanityCheck:
      with self.lock:
        self.results[request.id] = IPCResponse.Idle(request.id)
      return IPCOperationResponseData.Empty()
    
    if request.type == IPCOperationRequestType.TaskState:
      return IPCOperationResponseData.TaskState(results=self.results)
    
    
    
    raise Exception(f"Invalid IPC operation request type: {request.type}")

  def on_received_message(self, data: Any):
    try:
      operation = IPCOperationRequestWrapper.model_validate(data)
      return self.on_received_operation(operation.root)
    except ValidationError as e:
      # Not an operation request, pass.
      pass

    try:
      request = IPCRequestWrapper.model_validate(data)
    except ValidationError as e:
      logger.error(f"Failed to parse {data} into a proper IPC request.")
      return
    return self.on_received_request(request.root)


  def initialize(
    self,
    *,
    pool: concurrent.futures.ThreadPoolExecutor,
    handlers: dict[IPCRequestType, IPCTaskHandlerFn],
    channel: IPCChannel,
    backchannel: IPCChannel
  ) :
    self.pool = pool
    self.results = {}
    self.handlers = handlers
    self.channel = channel
    self.listener = IPCListener(backchannel, self.on_received_message)
    logger.info("Initialized IPCTaskReceiver.")

  def listen(self, stop_event: threading.Event):
    thread = threading.Thread(target=self.listener.listen, args=(stop_event,), daemon=True)
    thread.start()
    return thread
  
  def result(self, id: str)->Optional[IPCResponse]:
    return self.results.get(id, None)

class IPCTaskClient(metaclass=Singleton):
  channel: IPCChannel

  def request(self, msg: IPCRequest):
    client = IPCClient(self.channel)
    return client.send(msg)
  
  def operation(self, msg: IPCOperationRequest):
    client = IPCClient(self.channel)
    return client.send(msg)

  def result(self, id: str)->Optional[IPCResponse]:
    client = IPCClient(self.channel)
    raw_response = client.send(IPCOperationRequestData.GetResult(id=id))
    response = IPCOperationResponseWrapper.model_validate(raw_response)

    return cast(IPCOperationResponseData.Result, response.root).data
    
  def has_pending_task(self, id: str)->bool:
    task = self.result(id)
    if task is None:
      return False
    return task.status == IPCResponseStatus.Pending or task.status == IPCResponseStatus.Idle

  def initialize(self, *, channel: IPCChannel):
    self.channel = channel

