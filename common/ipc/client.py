from dataclasses import dataclass
import multiprocessing
from multiprocessing.connection import Client, Listener
import multiprocessing.connection
import multiprocessing.synchronize
from typing import Any, Callable
import concurrent

import pydantic

from common.logger import RegisteredLogger


logger = RegisteredLogger().provision("IPC")

class IPCChannel(pydantic.BaseModel):
  channel: tuple[str, int]
  authkey: bytes

SERVER2TOPIC_IPC_CHANNEL = IPCChannel(
  channel=("localhost", 12520),
  authkey=b"wordsmith"
)
TOPIC2SERVER_IPC_CHANNEL = IPCChannel(
  channel=("localhost", 12521),
  authkey=b"wordsmith"
)

class IPCClient:
  channel: IPCChannel
  def __init__(self, channel: IPCChannel) -> None:
    self.channel = channel

  def send(self, message: pydantic.BaseModel):
    try:
      client = Client(self.channel.channel, authkey=self.channel.authkey)
    except Exception as e:
      logger.error(f"Failed to initialize a connection with {self.channel.channel}. Error: {e}")
      raise e
    try:
      client.send(message)
      logger.info(f"Sent message with payload: {message.model_dump_json()}")
    except Exception as e:
      logger.error(f"An error occurred while sending the message {message.model_dump_json()} through IPC. Error: {e}")
      raise e

class IPCListener:
  channel: IPCChannel
  listener: Listener
  running: bool
  handler: Callable[[Any], None]

  def __init__(self, channel: IPCChannel, handler: Callable[[Any], None]) -> None:
    self.listener = Listener(channel.channel, authkey=channel.authkey)
    self.handler = handler
    self.running = False
    self.channel = channel

  def listen(self):
    self.running = True
    while self.running:
      try:
        conn = self.listener.accept()
      except Exception as e:
        logger.error(f"An error occurred while waiting for connection. Error: {e}")
        continue
      logger.info(f"Successfully established connection")
      
      while self.running:
        if not conn.poll():
          continue
        try:
          msg = conn.recv()
          logger.debug(f"Received message {msg} from the connection.")
        except EOFError as e:
          logger.info("The connection has closed. Waiting for new connection...")
          break
        except Exception as e:
          logger.error(f"Failed to receive a message from the connection. The connection will be aborted. Error: {e}")
          break

        self.handler(msg)
      conn.close()

@dataclass
class IntraProcessCommunicator:
  lock: multiprocessing.synchronize.Lock
  pipe: multiprocessing.connection.PipeConnection
  stop_event: multiprocessing.synchronize.Event

__all__ = [
  "IPCListener",
  "IPCClient"
]
