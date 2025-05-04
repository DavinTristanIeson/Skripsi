from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import threading
from typing import Any, Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.base import JobLookupError

from modules.baseclass import Singleton
from modules.logger import ProvisionedLogger
from modules.task.engine import scheduler
from .responses import TaskLog, TaskResponse, TaskStatusEnum


logger = ProvisionedLogger().provision("Task")

class TaskStopException(Exception):
  pass

@dataclass
class TaskStorageProxy:
  id: str
  response: TaskResponse
  stop_event: threading.Event
  
  def check_stop(self):
    if self.stop_event.is_set():
      raise TaskStopException()

  def log(self, message: str, status: TaskStatusEnum):
    logger.info(f"({status}) {message}")
    self.response.logs.append(TaskLog(
      message=message,
      status=status,
    ))
    self.check_stop()

  def log_success(self, message: str):
    self.log(message, TaskStatusEnum.Success)

  def log_error(self, message: str):
    self.log(message, TaskStatusEnum.Failed)
  
  def log_pending(self, message: str):
    self.log(message, TaskStatusEnum.Pending)
    
  def success(self, data: Any):
    logger.info(f"TASK {self.id} SUCCESS: {data}")
    self.response.status = TaskStatusEnum.Success
    self.response.data = data


@contextmanager
def task_execution_context(proxy: TaskStorageProxy):
  proxy.response.status = TaskStatusEnum.Pending
  try:
    yield proxy
  except TaskStopException as e:
    logger.warning(f"Task {proxy.id} has been cancelled successfully.")
    proxy.response.logs.append(TaskLog(
      message="Task has been cancelled.",
      status=TaskStatusEnum.Failed,
    ))
    proxy.response.status = TaskStatusEnum.Failed
  except Exception as e:
    logger.exception(e)
    proxy.response.status = TaskStatusEnum.Failed

TaskHandlerFn = Callable[[TaskStorageProxy], None]

class TaskConflictResolutionBehavior(str, Enum):
  Queue = "queue"
  Cancel = "cancel"
  Ignore = "ignore"

@dataclass
class AlternativeTaskResponse:
  data: Any
  message: str

class TaskStorage(metaclass=Singleton):
  results: dict[str, TaskResponse]
  stop_events: dict[str, threading.Event]
  lock: threading.Lock
  def __init__(self) -> None:
    self.results = {}
    self.lock = threading.Lock()

  def get_proxy(self, task_id: str):
    response = self.results.get(task_id, None)
    if response is None:
      response = TaskResponse.Idle(task_id)
      self.results[task_id] = response
    stop_event = self.stop_events.get(task_id, None)
    if stop_event is None:
      stop_event = threading.Event()
      self.stop_events[task_id] = stop_event
    return TaskStorageProxy(
      id=task_id,
      response=response,
      stop_event=stop_event,
    )
  
  def get_task_status(self, task_id: str)->Optional[TaskStatusEnum]:
    result = self.results.get(task_id, None)
    if result is None:
      return None
    return result.status
  
  def __invalidate_singular(self, task_id: str):
    stop_event = self.stop_events.get(task_id, None)
    if stop_event is not None:
      stop_event.set()
      self.stop_events.pop(task_id)

    if task_id in self.results:
      # Remove task
      self.results.pop(task_id)
    
    has_apscheduler_job = scheduler.get_job(task_id) is not None
    if has_apscheduler_job:
      try:
        scheduler.remove_job(task_id)
      except JobLookupError:
        pass
    logger.warning(f"Task {task_id} has been invalidated")

  
  def invalidate(
    self, *,
    task_id: Optional[str] = None,
    prefix: Optional[str] = None
  ):
    with self.lock:
      if task_id is not None:
        logger.warning(f"Requesting the invalidation of task {task_id}")
        self.__invalidate_singular(task_id)

      if prefix is not None:
        logger.warning(f"Requesting the invalidation of task with prefix {prefix}")
        affected_task_ids = filter(lambda key: key.startswith(prefix), self.stop_events.keys())
        for task_id in affected_task_ids:
          self.__invalidate_singular(task_id)
        
  def has_running_task(self, task_id: str):
    status = self.get_task_status(task_id)
    has_existing_job = status is not None and status.is_running
    return has_existing_job
  
  def proxy_context(self, task_id: str):
    return task_execution_context(self.get_proxy(task_id))
  
  def add_task(
    self, *,
    scheduler: AsyncIOScheduler,
    task_id: str, task: Callable[[Any], Any],
    args: list[Any], idle_message: str,
    conflict_resolution: TaskConflictResolutionBehavior
  ):
    # If status is not None, the task exists.
    has_existing_job = self.has_running_task(task_id)
    if has_existing_job:
      if conflict_resolution == TaskConflictResolutionBehavior.Ignore:
        # Don't add task.
        logger.warning(f"Skipping the re-execution of {task_id} since there is a task that is still running.")
        return
      if conflict_resolution == TaskConflictResolutionBehavior.Cancel:
        # Cancel existing task
        self.invalidate(task_id=task_id)

    scheduler.add_job(
      task,
      args=args,
      # misfire_grace_time prevents the jobs from being canceled
      # https://stackoverflow.com/questions/65690003/how-to-manage-a-task-queue-using-apscheduler
      id=task_id,
      misfire_grace_time=None,
    )
    store = TaskStorage()
    with store.lock:
      response = TaskResponse(
        id=task_id,
        logs=[
          TaskLog(
            status=TaskStatusEnum.Idle,
            message=idle_message,
          )
        ],
        data=None,
        status=TaskStatusEnum.Idle
      )
      store.results[task_id] = response
      store.stop_events[task_id] = threading.Event()
    logger.warning(f"Task {task_id} has been added.")

  def get_task_result(self, task_id: str, alternative_response: Optional[Callable[[], Optional[AlternativeTaskResponse]]]):
    with self.lock:
      response = self.results.get(task_id, None)
    
    # Use actual result
    if response is not None:
      return response
    
    # No alternatives. We can exit early.
    if alternative_response is None:
      return None
      
    alt_response = alternative_response()
    # No alternative response. We can exit early.
    if alt_response is None:
      return None
    
    response = TaskResponse(
      id=task_id,
      logs=[
        TaskLog(
          status=TaskStatusEnum.Success,
          message=alt_response.message,
        )
      ],
      data=alt_response.data,
      status=TaskStatusEnum.Success
    )
    with self.lock:
      self.results[task_id] = response
    # Return alternative response
    return response

__all__ = [
  "TaskStorageProxy",
  "AlternativeTaskResponse",
  "TaskConflictResolutionBehavior"
]