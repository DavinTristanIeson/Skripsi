import concurrent.futures
import threading
from typing import Any, Callable

import concurrent
from common.ipc.client import IPCChannel, IPCClient, IPCListener
from common.ipc.requests import IPCRequest, IPCRequestType, IPCRequestWrapper
from common.ipc.responses import IPCResponse
from common.logger import RegisteredLogger
from common.models.metaclass import Singleton

logger = RegisteredLogger().provision("IPC")

class IPCTaskReceiver(metaclass=Singleton):
  pool: concurrent.futures.ProcessPoolExecutor
  handlers: dict[IPCRequestType, Callable[[IPCRequest], None]]
  listener: IPCListener
  
  def __init__(self):
    pass

  def on_received_message(self, data: Any):
    try:
      response = IPCRequestWrapper.model_validate(data)
    except Exception as e:
      logger.error(f"Failed to parse {data} into a proper IPC request.")
      return
    
    handler = self.handlers.get(response.root.type, None)
    if handler is None:
      logger.error(f"No handler for message of type {response.root.type} has been registered!")
      return
    handler(response.root)

  def initialize(
    self,
    *,
    pool: concurrent.futures.ProcessPoolExecutor,
    handlers: dict[IPCRequestType, Callable[[IPCRequest], None]],
    backchannel: IPCChannel
  ) :
    self.pool = pool
    self.handlers = handlers
    self.listener = IPCListener(backchannel, self.on_received_message)

  def listen(self):
    thread = threading.Thread(target=self.listener.listen, daemon=True)
    thread.start()
    return thread


class IPCTaskLocker(metaclass=Singleton):
  result: dict[str, IPCResponse]
  listener: IPCListener
  client: IPCClient

  def on_received_message(self, data: Any):
    try:
      response = IPCResponse.model_validate(data)
    except Exception as e:
      logger.error(f"Failed to parse {data} into a proper IPC response.")
      return
    
    logger.info(f"Task {response.id} has been updated with status: {response.status.upper()}{' . Error: ' + response.error if response.error is not None else ''}.")
    self.result[response.id] = response

  def request(self, msg: IPCRequest):
    self.client.send(msg)

  def initialize(self, *, channel: IPCChannel, backchannel: IPCChannel):
    self.result = dict()
    self.client = IPCClient(channel)
    self.listener = IPCListener(backchannel, self.on_received_message)

  def listen(self):
    thread = threading.Thread(target=self.listener.listen, daemon=True)
    thread.start()
    return thread

