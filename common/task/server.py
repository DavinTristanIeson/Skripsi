import concurrent.futures
import queue
import threading
import traceback
from typing import Any, Optional, cast

from pydantic import ValidationError

from common.logger import RegisteredLogger, TimeLogger
from common.models.api import ApiError
from common.models.metaclass import Singleton
from .requests import TaskRequest, TaskRequestType
from .executor import TaskHandlerFn, TaskPayload
from .responses import TaskResponse

logger = RegisteredLogger().provision("Task")

class TaskServer(metaclass=Singleton):
  pool: concurrent.futures.ThreadPoolExecutor
  handlers: dict[TaskRequestType, TaskHandlerFn]
  results: dict[str, TaskResponse]
  lock: threading.Lock
  ongoing_tasks: dict[str, threading.Event]

  def __init__(self) -> None:
    super().__init__()
    self.ongoing_tasks = {}
    self.lock = threading.Lock()

  def handle_task(self, handler: TaskHandlerFn, message: TaskRequest):
    stop_event = threading.Event()
    with self.lock:
      self.ongoing_tasks[message.id] = stop_event
    with TimeLogger(logger, f"Handling task {message.id} with payload: {message.model_dump_json()}"):
      try:
        handler(
          TaskPayload(
            id=message.id,
            lock=self.lock,
            results=self.results,
            stop_event=stop_event,
            request=message
          ),
        )
      except Exception as e:
        logger.error(f"An error has occurred during the execution of task {message.id}. Error: {traceback.print_exception(e)}")
        with self.lock:
          self.results[message.id] = TaskResponse.Error(message.id, str(e))
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
    
  def begin_task(self, request: TaskRequest):
    handler = self.handlers.get(request.data.type, None)
    if handler is None:
      logger.error(f"No handler for message of type {request.data.type} has been registered!")
      return
    
    with self.lock:
      has_task_id = request.id in self.ongoing_tasks
    if has_task_id:
      logger.warning(f"Canceling ongoing task for {request.id} in lieu of the new task: {request.model_dump_json()}")
      self.cancel_task(request.id)

    with self.lock:
      response = TaskResponse.Idle(request.id)
      self.results[request.id] = response

    self.pool.submit(self.handle_task, handler, request)
  
  def result(self, id: str)->Optional[TaskResponse]:
    return self.results.get(id, None)
  
  def sanity_check(self, id: str)->Optional[TaskResponse]:
    sanity = TaskResponse.Idle(id)
    with self.lock:
      self.results[id] = sanity
  
  def clear_tasks(self, prefix: Optional[str] = None):
    with self.lock:
      to_be_removed = filter(lambda x: prefix is None or x.startswith(prefix), list(self.results.keys()))
      for id in to_be_removed:
        if id in self.ongoing_tasks:
          self.ongoing_tasks[id].set()
          self.ongoing_tasks.pop(id)
        self.results.pop(id)
        
  def initialize(
    self,
    *,
    pool: concurrent.futures.ThreadPoolExecutor,
    handlers: dict[TaskRequestType, TaskHandlerFn],
  ):
    self.pool = pool
    self.results = {}
    self.handlers = handlers
    logger.info("Initialized TaskServer.")
