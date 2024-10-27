import os
from fastapi import APIRouter, Query

from common.ipc.requests import IPCRequestData
from common.ipc.responses import IPCResponse, IPCResponseData
from common.ipc.taskqueue import IPCTaskClient
from common.models.api import ApiError, ApiResult
from server.controllers.project_checks import ProjectExistsDependency, PerformedTopicModelingDependency, SchemaColumnExistsDependency
from wordsmith.data.paths import ProjectPaths

router = APIRouter(
  tags=["Topics"]
)

@router.post('/{project_id}/topics/start')
def post__topic_modeling_request(config: ProjectExistsDependency, project_id: str):
  locker = IPCTaskClient()

  task_id = IPCRequestData.TopicModeling.task_id(project_id)
  has_pending_task = locker.has_pending_task(task_id)

  # Always cancel old task
  IPCTaskClient().request(IPCRequestData.TopicModeling(
    id=task_id,
    project_id=project_id
  ))

  if has_pending_task:
    return ApiResult(data=None, message=f"The topic modeling algorithm will soon be applied to Project \"{project_id}\". Please wait for a few seconds (or minutes depending on the size of your dataset) for the algorithm to complete.")
  else:
    return ApiResult(data=None, message=f"The topic modeling algorithm will be applied again to Project \"{project_id}\"; meanwhile, the previous pending task will be canceled. Please wait for a few seconds (or minutes depending on the size of your dataset) for the algorithm to complete.")


@router.get('/{project_id}/topics/status')
def get__topic_modeling_status(config: ProjectExistsDependency, project_id: str):
  bertopic_path = config.paths.full_path(ProjectPaths.BERTopic)
  task_id = IPCRequestData.TopicModeling.task_id(project_id)
  locker = IPCTaskClient()
  if result:=locker.result(task_id):
    return result
  
  if os.path.exists(bertopic_path):
    return IPCResponse.Success(task_id, IPCResponseData.Empty(), message="The topic modeling procedure has already been executed before. Feel free to explore the discovered topics.")
  
  raise ApiError(f"No topic modeling task has been started for {project_id}.", 400)

@router.get('/{project_id}/topics/{column}')
def get__topics(task: PerformedTopicModelingDependency, project_id: str, col: SchemaColumnExistsDependency):
  client = IPCTaskClient()

  task_id = IPCRequestData.Topics.task_id(project_id)
  if result:=client.result(task_id):
    return result

  raise ApiError(f"No topic has been requested for {project_id} > {col.name}.", 400)

@router.post('/{project_id}/topics/{column}')
def post__topics(task: PerformedTopicModelingDependency, project_id: str, col: SchemaColumnExistsDependency):
  IPCTaskClient().request(IPCRequestData.Topics(
    id=project_id,
    project_id=project_id,
    column=col.name,
  ))
  return ApiResult(data=None, message=f"Please wait for a few seconds while we create the visualizations for the topics in {col.name} from {project_id}")


@router.get('/{project_id}/topics/{column}/similarity')
def get__topic_similarity(task: PerformedTopicModelingDependency, project_id: str, col: SchemaColumnExistsDependency):
  locker = IPCTaskClient()

  task_id = IPCRequestData.TopicSimilarityPlot.task_id(project_id)
  if result:=locker.result(task_id):
    return result
  
  IPCTaskClient().request(IPCRequestData.TopicSimilarityPlot(
    id=task_id,
    project_id=project_id,
    column=col.name,
  ))
  
  return ApiResult(data=None, message=f"Please wait for a few seconds while we calculate the similarities for the topics in {col.name} from {project_id}")