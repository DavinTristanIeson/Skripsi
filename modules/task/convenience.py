from dataclasses import dataclass
from typing import Any, Callable, Optional

from modules.task.manager import TaskManager
from modules.task.responses import TaskLog, TaskResponse, TaskStatusEnum

@dataclass
class AlternativeTaskResponse:
  data: Any
  message: str

def get_task_result_or_else(task_id: str, alternative_response: Optional[Callable[[], Optional[AlternativeTaskResponse]]]):
  taskmanager = TaskManager()
  with taskmanager.lock:
    response = taskmanager.results.get(task_id, None)
  
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
  with taskmanager.lock:
    taskmanager.results[task_id] = response
  # Return alternative response
  return response

__all__ = [
  "AlternativeTaskResponse",
  "get_task_result_or_else",
]