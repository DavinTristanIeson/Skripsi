from enum import Enum
import logging
from multiprocessing.connection import Client, Connection, Listener
from types import SimpleNamespace
from typing import Annotated, Callable, Literal, Union
import pydantic

from common.metaclass import Singleton

class IPCMessageType(str, Enum):
  TopicModelingRequest = "topic_modeling_request"
  TopicHierarchyPlotRequest = "topic_hierarchy_plot_request"

class IPCMessageVariant(SimpleNamespace):
  class TopicModelingRequest(pydantic.BaseModel):
    type: Literal[IPCMessageType.TopicModelingRequest] = IPCMessageType.TopicModelingRequest
    id: str

  class TopicHierarchyPlotRequest(pydantic.BaseModel):
    type: Literal[IPCMessageType.TopicHierarchyPlotRequest] = IPCMessageType.TopicHierarchyPlotRequest
    id: str
  

IPCMessage = Union[
  IPCMessageVariant.TopicModelingRequest,
  IPCMessageVariant.TopicHierarchyPlotRequest
]
class IPCMessageWrapper(pydantic.RootModel):
  root: Annotated[
  IPCMessage,
  pydantic.Field(discriminator="type")
]

IPC_CHANNEL = ("localhost", 5500)
IPC_AUTH_KEY = b"wordsmith"


logger = logging.getLogger("IPC")
class IPCClient(metaclass=Singleton):
  client: Connection
  def __init__(self) -> None:
    self.client = Client(IPC_CHANNEL, authkey=IPC_AUTH_KEY)

  def send(self, message: IPCMessage):
    if self.client.closed:
      self.client = Client(IPC_CHANNEL, authkey=IPC_AUTH_KEY)
    try:
      self.client.send(message)
      logger.info(f"Sent message with payload: {message.model_dump_json()}")
    except Exception as e:
      logger.error(f"An error occurred while sending the message {message.model_dump_json()} through IPC. Error: {e}")

class IPCListener(metaclass=Singleton):
  listener: Listener
  handlers: dict[IPCMessageType, Callable[[IPCMessage], None]]
  running: bool
  def __init__(self, handlers: dict[IPCMessageType, Callable[[IPCMessage], None]]) -> None:
    self.listener = Listener(IPC_CHANNEL, authkey=IPC_AUTH_KEY)
    self.handlers = handlers
    self.running = False

  def listen(self):
    self.running = True
    while self.running:
      conn = self.listener.accept()
      logger.info(f"Successfully established connection")
      while self.running and not conn.closed:
        try:
          msg = conn.recv()
        except EOFError as e:
          logger.info("Shutting down application as the connection has closed.")
          self.running = False
          return
        except Exception as e:
          logger.error(f"Failed to receive a message from the connection. Error: {e}")
          continue

        try:
          message = IPCMessageWrapper.model_validate(msg)
        except Exception as e:
          logger.error(f"Failed to parse message {msg} into a proper IPCMessage. Error: {e}")
          continue
        handler = self.handlers.get(message.root.type, None)
        if handler is not None:
          handler(message.root)
        else:
          logger.error(f"No handler has been registered for IPC message of type {message.root.type.value}!")
      conn.close()


__all__ = [
  "IPCMessage",
  "IPCMessageVariant",
  "IPCMessageType",
  "IPC_CHANNEL",
  "IPCListener",
  "IPCClient"
]
