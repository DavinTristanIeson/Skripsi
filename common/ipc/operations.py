import datetime
from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Literal, Optional, Sequence, Union

import pydantic

from common.ipc.responses import IPCResponse


class IPCOperationRequestType(str, Enum):
  CancelTask = "cancel_task"
  GetResult = "get_task"
  TaskState = "task_state"
  SanityCheck = "sanity_check"



class IPCOperationRequestData(SimpleNamespace):
  class CancelTask(pydantic.BaseModel):
    type: Literal[IPCOperationRequestType.CancelTask] = IPCOperationRequestType.CancelTask
    id: str
  class GetResult(pydantic.BaseModel):
    type: Literal[IPCOperationRequestType.GetResult] = IPCOperationRequestType.GetResult
    id: str
  class SanityCheck(pydantic.BaseModel):
    type: Literal[IPCOperationRequestType.SanityCheck] = IPCOperationRequestType.SanityCheck
    id: str
  class TaskState(pydantic.BaseModel):
    type: Literal[IPCOperationRequestType.TaskState] = IPCOperationRequestType.TaskState

  TypeUnion = Union[CancelTask, GetResult, SanityCheck, TaskState]
  DiscriminatedUnion = Annotated[TypeUnion, pydantic.Field(discriminator="type")]
    

class IPCOperationResponseType(str, Enum):
  Result = "result"
  Error = "error"
  Empty = "empty"
  TaskState = "task_state"

class IPCOperationResponseData(SimpleNamespace):
  class Result(pydantic.BaseModel):
    type: Literal[IPCOperationResponseType.Result] = IPCOperationResponseType.Result
    data: Optional[IPCResponse]
  class Error(pydantic.BaseModel):
    type: Literal[IPCOperationResponseType.Error] = IPCOperationResponseType.Error
    error: str
  class Empty(pydantic.BaseModel):
    type: Literal[IPCOperationResponseType.Empty] = IPCOperationResponseType.Empty
  class TaskState(pydantic.BaseModel):
    type: Literal[IPCOperationResponseType.TaskState] = IPCOperationResponseType.TaskState
    results: dict[str, IPCResponse]
  
  TypeUnion = Union[Result, Empty, TaskState, Error]

IPCOperationRequest = Annotated[IPCOperationRequestData.TypeUnion, pydantic.Field(discriminator="type")]
IPCOperationResponse = Annotated[IPCOperationResponseData.TypeUnion, pydantic.Field(discriminator="type")]
class IPCOperationRequestWrapper(pydantic.RootModel):
  root: IPCOperationRequest

class IPCOperationResponseWrapper(pydantic.RootModel):
  root: IPCOperationResponse