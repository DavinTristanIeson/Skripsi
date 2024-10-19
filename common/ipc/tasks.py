import concurrent.futures
import functools
import multiprocessing
import multiprocessing.synchronize
import threading
from typing import Any, Callable, Generator, Iterable, Union, cast

import concurrent
from common.ipc.client import IPCChannel, IPCClient, IPCListener, IntraProcessCommunicator
from common.ipc.requests import IPCRequest, IPCRequestData, IPCRequestType, IPCRequestWrapper
from common.ipc.responses import IPCProgressReport, IPCResponse, IPCResponseData, IPCResponseDataUnion, IPCResponseStatus
from common.logger import RegisteredLogger, TimeLogger
from common.models.api import ApiError
from common.models.metaclass import Singleton

logger = RegisteredLogger().provision("IPC")

IPCTaskHandlerFn = Callable[[IntraProcessCommunicator, IPCRequest], Generator[Union[IPCProgressReport, IPCResponseDataUnion]]]
IPCTaskEvent = multiprocessing.Event

@functools.wraps
def ipc_task_handler(fn: IPCTaskHandlerFn):
  def inner(comm: IntraProcessCommunicator, message: IPCRequest):
    try:
      for progress in fn(comm, message):
        if isinstance(progress, IPCProgressReport):
          comm.pipe.send(IPCResponse(
            data=IPCResponseData.Empty(),
            error=None,
            message=progress.message,
            progress=progress.progress,
            status=IPCResponseStatus.Pending,
            id=message.id,
          ))
        else:
          comm.pipe.send(IPCResponse(
            data=progress,
            error=None,
            message=None,
            progress=1,
            status=IPCResponseStatus.Pending,
            id=message.id,
          ))
    except ApiError as e:
      if comm.stop_event.is_set():
        return
      comm.pipe.send(IPCResponse(
        data=IPCResponseData.Empty(),
        error=e.message,
        message=None,
        progress=1,
        status=IPCResponseStatus.Failed,
        id=message.id,
      ))
    except Exception as e:
      if comm.stop_event.is_set():
        return
      comm.pipe.send(IPCResponse(
        data=IPCResponseData.Empty(),
        error=str(e),
        message=None,
        progress=1,
        status=IPCResponseStatus.Failed,
        id=message.id,
      ))
  return inner

class IPCTaskReceiver(metaclass=Singleton):
  pool: concurrent.futures.ProcessPoolExecutor
  listener_pool: concurrent.futures.ThreadPoolExecutor
  handlers: dict[IPCRequestType, IPCTaskHandlerFn]
  listener: IPCListener
  lock: multiprocessing.synchronize.Lock
  channel: IPCChannel

  ongoing_tasks: dict[str, multiprocessing.synchronize.Event]

  def respond(self, response: IPCResponse):
    client = IPCClient(self.channel)
    try:
      client.send(response)
    except Exception as e:
      logger.error(f"An unexpected error has occurred while sending {response.model_dump_json()} through the client. Error: {e}")

  def handle_task(self, handler: IPCTaskHandlerFn, message: IPCRequest):
    read_pipe, write_pipe = multiprocessing.Pipe()
    stop_event = multiprocessing.Event()

    with TimeLogger(logger, f"Handling task {message.id} with payload: {message.model_dump_json()}"):
      future = self.pool.submit(
        handler,
        IntraProcessCommunicator(
          lock=self.lock,
          pipe=write_pipe,
          stop_event=stop_event
        ),
        message
      )
      self.ongoing_tasks[message.id] = stop_event

      try:
        while not future.done():
          report = read_pipe.recv()
          if report is not None:
            self.respond(report)
      except Exception as e:
        logger.error(f"An unexpected error has occurred while waiting for task of {message.model_dump_json()} to finish. Error: {e}")
      
    if message.id in self.ongoing_tasks:
      self.ongoing_tasks.pop(message.id)
    else:
      logger.warning(f"The ongoing task for task {message.id} had been removed unexpectedly.")

  def cancel_task(self, id: str):
    if id not in self.ongoing_tasks:
      return
    self.ongoing_tasks[id].set()
    self.ongoing_tasks.pop(id)

  def on_received_message(self, data: Any):
    try:
      response = IPCRequestWrapper.model_validate(data)
    except Exception as e:
      logger.error(f"Failed to parse {data} into a proper IPC request.")
      return
    
    if response.root.type == IPCRequestType.CancelTask:
      pass
    
    handler = self.handlers.get(response.root.type, None)
    if handler is None:
      logger.error(f"No handler for message of type {response.root.type} has been registered!")
      return
    if response.root.id in self.ongoing_tasks:
      logger.warning(f"Canceling ongoing task for {response.root.id} in lieu of the new task: {response.root.model_dump_json()}")
      self.cancel_task(response.root.id)

    self.listener_pool.submit(self.handle_task, handler, response.root)

  def initialize(
    self,
    *,
    pool: concurrent.futures.ProcessPoolExecutor,
    listener_pool: concurrent.futures.ThreadPoolExecutor,
    handlers: dict[IPCRequestType, IPCTaskHandlerFn],
    channel: IPCChannel,
    backchannel: IPCChannel
  ) :
    self.pool = pool
    self.listener_pool = listener_pool
    self.handlers = handlers
    self.channel = channel
    self.listener = IPCListener(backchannel, self.on_received_message)

  def listen(self):
    thread = threading.Thread(target=self.listener.listen)
    thread.start()
    return thread


class IPCTaskLocker(metaclass=Singleton):
  result: dict[str, IPCResponse]
  listener: IPCListener
  channel: IPCChannel

  def on_received_message(self, data: Any):
    try:
      response = IPCResponse.model_validate(data)
    except Exception as e:
      logger.error(f"Failed to parse {data} into a proper IPC response.")
      return
    
    logger.info(f"Task {response.id} has been updated with status: {response.status.upper()}{' . Error: ' + response.error if response.error is not None else ''}.")
    self.result[response.id] = response

  def request(self, msg: IPCRequest):
    client = IPCClient(self.channel)
    client.send(msg)

  def initialize(self, *, channel: IPCChannel, backchannel: IPCChannel):
    self.result = dict()
    self.channel = channel
    self.listener = IPCListener(backchannel, self.on_received_message)

  def listen(self):
    thread = threading.Thread(target=self.listener.listen, daemon=True)
    thread.start()
    return thread

