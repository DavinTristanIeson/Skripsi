import asyncio

from common.models.metaclass import Singleton

class TaskTracker(metaclass=Singleton):
  tasks: set[asyncio.Task]
  def __init__(self):
    self.tasks = set()
  def enqueue(self, coroutine)->asyncio.Task:
    task = asyncio.create_task(coroutine)
    self.tasks.add(task)
    task.add_done_callback(self.tasks.remove)

    return task

__all__ = [
  "TaskTracker"
]