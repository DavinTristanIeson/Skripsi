import concurrent.futures
import multiprocessing.synchronize
import queue
import threading
import time
from typing import Any, Optional

from common.ipc.client import IPCChannel, IPCClient, IPCListener
from common.ipc.requests import IPCRequest, IPCRequestType, IPCRequestWrapper
from common.ipc.responses import IPCResponse, IPCResponseData, IPCResponseStatus
from common.ipc.task import IPCTask, IPCTaskHandlerFn
from common.logger import RegisteredLogger, TimeLogger
from common.models.metaclass import Singleton

logger = RegisteredLogger().provision("IPC")

class IPCTaskReceiver(metaclass=Singleton):
  pool: concurrent.futures.ThreadPoolExecutor
  handlers: dict[IPCRequestType, IPCTaskHandlerFn]
  listener: IPCListener
  lock: threading.Lock
  channel: IPCChannel
  ongoing_tasks: dict[str, threading.Event]
  message_queue: queue.Queue[IPCResponse]

  def __init__(self) -> None:
    super().__init__()
    self.ongoing_tasks = {}
    self.lock = threading.Lock()
    self.message_queue = queue.Queue()

  def handle_task(self, handler: IPCTaskHandlerFn, message: IPCRequest):
    stop_event = threading.Event()
    with TimeLogger(logger, f"Handling task {message.id} with payload: {message.model_dump_json()}"):
      try:
        future = self.pool.submit(
          handler,
          IPCTask(
            id=message.id,
            lock=self.lock,
            pipe=self.message_queue,
            stop_event=stop_event,
            request=message
          ),
        )
        self.ongoing_tasks[message.id] = stop_event
        future.result()
      except Exception as e:
        logger.error(f"An error has occurred during the execution of task {message.id}. Error: {e}")
        self.message_queue.put(IPCResponse.Error(message.id, str(e)))
        return
      
    if message.id in self.ongoing_tasks:
      self.ongoing_tasks.pop(message.id)
      logger.debug(f"Cleaned up task for {message.id}.")
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

    self.pool.submit(self.handle_task, handler, response.root)

  def initialize(
    self,
    *,
    pool: concurrent.futures.ThreadPoolExecutor,
    handlers: dict[IPCRequestType, IPCTaskHandlerFn],
    channel: IPCChannel,
    backchannel: IPCChannel
  ) :
    self.pool = pool
    self.handlers = handlers
    self.channel = channel
    self.listener = IPCListener(backchannel, self.on_received_message)
    logger.info("Initialized IPCTaskReceiver.")



  def send_message(self, client: IPCClient):
    if self.message_queue.empty():
      return
    response = self.message_queue.get()
    try:
      client.send(response)
      logger.debug(f"Sent {response.model_dump_json()} to {self.channel}")
    except Exception as e:
      logger.error(f"An unexpected error has occurred while sending {response.model_dump_json()} to {self.channel}. Error: {e}")

  def process_message_queue(self, stop_event: threading.Event):
    while not stop_event.is_set():
      try:
        client = IPCClient(self.channel)
        while not stop_event.is_set():
          self.send_message(client)
          
      except ConnectionError:
        logger.error(f"Failed to connect to {self.channel}. Trying again in 3 seconds.")
        time.sleep(3)
        pass
      except Exception as e:
        logger.error(f"An unexpected error has occurred while sending a message through the client. Error: {e}")

  def listen(self, stop_event: threading.Event):
    thread1 = threading.Thread(target=self.listener.listen, args=(stop_event,), daemon=True)
    thread2 = threading.Thread(target=self.process_message_queue, args=(stop_event,), daemon=True)
    thread1.start()
    thread2.start()
    return thread1, thread2


class IPCTaskLocker(metaclass=Singleton):
  results: dict[str, IPCResponse]
  listener: IPCListener
  channel: IPCChannel

  def on_received_message(self, data: Any):
    try:
      response = IPCResponse.model_validate(data)
    except Exception as e:
      logger.error(f"Failed to parse {data} into a proper IPC response.")
      return
    
    logger.info(f"Task {response.id} has been updated with status: {response.status.upper()}{' . Error: ' + response.error if response.error is not None else ''}.")
    self.results[response.id] = response

  def request(self, msg: IPCRequest):
    client = IPCClient(self.channel)
    client.send(msg)
    self.results[msg.id] = IPCResponse.Idle(msg.id)

  def initialize(self, *, channel: IPCChannel, backchannel: IPCChannel):
    self.results = dict()
    self.channel = channel
    self.listener = IPCListener(backchannel, self.on_received_message)

  def result(self, id: str)->Optional[IPCResponse]:
    return self.results.get(id, None)
  
  def has_pending_task(self, id: str)->bool:
    task = self.results.get(id, None)
    if task is None:
      return False
    return task.status == IPCResponseStatus.Pending or task.status == IPCResponseStatus.Idle

  def listen(self, stop_event: threading.Event):
    thread = threading.Thread(target=self.listener.listen, args=(stop_event,), daemon=True)
    thread.start()
    return thread

