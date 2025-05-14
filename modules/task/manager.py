from contextlib import contextmanager
from enum import Enum
import multiprocessing
import multiprocessing.managers
from queue import Queue
import queue
import threading
from typing import Any, Callable, Optional

from apscheduler.jobstores.base import JobLookupError
from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore

from modules.baseclass import Singleton
from modules.logger import ProvisionedLogger
from modules.task.proxy import TaskManagerProxy
from .responses import TaskLog, TaskResponse, TaskStatusEnum


logger = ProvisionedLogger().provision("Task")
# Register apscheduler logger
ProvisionedLogger().provision("apscheduler")

class TaskConflictResolutionBehavior(str, Enum):
  Queue = "queue"
  Cancel = "cancel"
  Ignore = "ignore"

class TaskManager(metaclass=Singleton):
  results: dict[str, TaskResponse]
  stop_events: dict[str, threading.Event]
  lock: threading.RLock
  queue: Queue
  manager: multiprocessing.managers.SyncManager
  scheduler: AsyncIOScheduler

  def __init__(self) -> None:
    self.results = {}
    self.stop_events = {}
    self.manager = multiprocessing.Manager()
    self.queue = self.manager.Queue()
    self.lock = threading.RLock()
    self.scheduler = AsyncIOScheduler(
      executors=dict(
        # 2 subprocesses to run tasks in parallel.
        default=ProcessPoolExecutor(
          max_workers=2,
        )
      ),
      jobstores=dict(
        default=MemoryJobStore(),
      ),
    )
    
  def get_task(self, task_id: str)->Optional[TaskResponse]:
    with self.lock:
      result = self.results.get(task_id, None)
    if result is None:
      return None
    return result
  
  def invalidate_task(self, task_id: str, *, clear: bool):
    with self.lock:
      stop_event = self.stop_events.get(task_id, None)
      if stop_event is not None:
        stop_event.set()
        self.stop_events.pop(task_id)
      
      has_apscheduler_job = self.scheduler.get_job(task_id) is not None
      if has_apscheduler_job:
        try:
          self.scheduler.remove_job(task_id)
        except JobLookupError:
          pass

      if clear:
        if task_id in self.results:
          # Remove task
          self.results.pop(task_id)
    logger.warning(f"Task {task_id} has been invalidated")

  def invalidate(
    self, *,
    clear: bool,
    all: bool = False,
    task_id: Optional[str] = None,
    prefix: Optional[str] = None
  ):
    with self.lock:
      if all:
        logger.warning(f"Requesting the invalidation of all tasks")
        all_tasks = self.results.keys()
        for event in all_tasks:
          self.invalidate_task(event, clear=clear)

      if task_id is not None:
        logger.warning(f"Requesting the invalidation of task {task_id}")
        self.invalidate_task(task_id, clear=clear)

      if prefix is not None:
        logger.warning(f"Requesting the invalidation of task with prefix {prefix}")
        affected_task_ids = list(filter(lambda key: key.startswith(prefix), self.results.keys()))
        for task_id in affected_task_ids:
          self.invalidate_task(task_id, clear=clear)
        
  def proxy(self, task_id: str)->TaskManagerProxy:
    with self.lock:
      response = self.results.get(task_id, None)
    if response is None:
      response = TaskResponse.Idle(task_id)
      self.results[task_id] = response
    stop_event = self.stop_events.get(task_id, None)
    if stop_event is None:
      stop_event = self.manager.Event()
      self.stop_events[task_id] = stop_event

    return TaskManagerProxy(
      id=task_id,
      queue=self.queue,
      stop_event=stop_event,
      response=response,
    )
    
  def add_task(
    self, *,
    task_id: str, task: Callable[[TaskManagerProxy, Any], Any],
    args: list[Any], idle_message: str,
    conflict_resolution: TaskConflictResolutionBehavior
  ):
    # If status is not None, the task exists.
    task_result = self.get_task(task_id)
    has_existing_job = task_result is not None and (task_result == TaskStatusEnum.Idle or task_result == TaskStatusEnum.Pending)
    if has_existing_job:
      if conflict_resolution == TaskConflictResolutionBehavior.Ignore:
        # Don't add task.
        logger.warning(f"Skipping the re-execution of {task_id} since there is a task that is still running.")
        return
      if conflict_resolution == TaskConflictResolutionBehavior.Cancel:
        # Cancel existing task
        self.invalidate(task_id=task_id, clear=True)

    with self.lock:
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
      self.results[task_id] = response
      self.stop_events[task_id] = self.manager.Event() 
      self.scheduler.add_job(
        task,
        # Insert the proxy as the first argument.
        # DON'T LET THE CALLER CREATE THE PROXY. IT IS VERY LIKELY THAT THE CALLER WILL CREATE A PROXY WITH AN OUTDATED RESPONSE
        # (especially after we've invalidated the response)
        args=[self.proxy(task_id), *args],
        # misfire_grace_time prevents the jobs from being canceled
        # https://stackoverflow.com/questions/65690003/how-to-manage-a-task-queue-using-apscheduler
        id=task_id,
        coalesce=True,
        misfire_grace_time=None,
        max_instances=1,
      )
    logger.warning(f"Task {task_id} has been added.")

  
  def receive_task_response(self, stop_event: threading.Event):
    logger.info(f"Task response queue is ready.")
    while not stop_event.is_set():
      try:
        response: TaskResponse = self.queue.get(timeout=2)
      except (TimeoutError, queue.Empty, queue.Full):
        continue
      with self.lock:
        logger.debug(f"Task response queue received response for task {response.id}.")
        if response.id not in self.results:
          # Response is already invalidated or has never been started. This may be a response from a lingering process.
          logger.debug(f"Ignored response for task {response.id}.")
          return
        self.results[response.id] = response
  
  @contextmanager
  def run(self):
    self.scheduler.start()
    receiver_thread_stop_event = threading.Event()
    try:
      receiver_thread = threading.Thread(target=self.receive_task_response, name="Receive Task Response", args=(receiver_thread_stop_event,))
      receiver_thread.start()
      yield
    except Exception as e:
      pass
    logger.info(f"Shutting down task response queue and task scheduler...")
    receiver_thread_stop_event.set()
    with self.lock:
      for event in self.stop_events.values():
        event.set()
    self.scheduler.shutdown()



__all__ = [
  "TaskManagerProxy",
  "TaskConflictResolutionBehavior",
]