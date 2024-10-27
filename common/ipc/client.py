from dataclasses import dataclass
import multiprocessing
from multiprocessing.connection import Client, Listener
import multiprocessing.connection
import multiprocessing.synchronize
import queue
import threading
from typing import Any, Callable

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
      conn = Client(self.channel.channel, authkey=self.channel.authkey)
    except Exception as e:
      logger.error(f"Failed to initialize a connection with {self.channel.channel}. Error: {e}")
      raise e
    try:
      conn.send(message)
      logger.debug(f"Sent message with payload: {message.model_dump_json()}")
    except Exception as e:
      logger.error(f"An error occurred while sending the message {message.model_dump_json()} through IPC. Error: {e}")
      raise e
    
    try:
      if conn.poll(timeout=2000):
        data = conn.recv()
        logger.debug(f"Received message with payload: {message.model_dump_json()}")
        return data
      else:
        raise Exception(f"Failed to receive any response from {self.channel.channel}. The IPC listener may have been shut down.")
      
    except Exception as e:
      logger.error(f"An error occurred while waiting for a message from {self.channel.channel}. Error: {e}")
      conn.close()
      raise e
    

  
    

class IPCListener:
  channel: IPCChannel
  listener: Listener
  handler: Callable[[Any], Any]

  def __init__(self, channel: IPCChannel, handler: Callable[[Any], Any]) -> None:
    self.listener = Listener(channel.channel, authkey=channel.authkey)
    self.handler = handler
    self.channel = channel

  def listen(self, stop_event: threading.Event):
    while not stop_event.is_set():
      try:
        conn = self.listener.accept()
      except Exception as e:
        logger.error(f"An error occurred while waiting for connection. Error: {e}")
        continue
      logger.info(f"Successfully established connection")
      
      while not stop_event.is_set():
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
        try:
          reply = self.handler(msg)
        except Exception as e:
          logger.error(f"An unexpected error occurred while handling {msg}. Error: {e}")
          break

        try:
          msg = conn.send(reply)
          logger.debug(f"Replied to connection with message {reply}.")
        except Exception as e:
          logger.error(f"Failed to send a message to the connection. The connection will be aborted. Error: {e}")
          break
      conn.close()

__all__ = [
  "IPCListener",
  "IPCClient"
]
