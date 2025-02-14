import datetime
from typing import TYPE_CHECKING, Optional

import pydantic

from models.config.paths import ProjectPathManager, ProjectPaths


class TopicModel(pydantic.BaseModel):
  id: int
  words: list[tuple[str, float]]
  label: str
  frequency: int

class TopicHierarchyModel(pydantic.BaseModel):
  id: int
  words: list[tuple[str, float]]
  label: str
  frequency: int
  children: Optional[list["TopicHierarchyModel"]] = None

class TopicModelingResultModel(pydantic.BaseModel):
  project_id: str
  topics: list[TopicModel]
  hierarchy: TopicHierarchyModel
  frequency: int
  created_at: datetime.datetime = pydantic.Field(
    default_factory=lambda: datetime.datetime.now()
  )

  def save_as_json(self, column: str):
    paths = ProjectPathManager(project_id=self.project_id)
    topics_path = paths.allocate_path(ProjectPaths.Topics(column))
    with open(topics_path, 'w', encoding='utf-8') as f:
      f.write(self.model_dump_json(indent=4))

  @staticmethod
  def load(project_id: str, column: str)->"TopicModelingResultModel":
    import orjson

    paths = ProjectPathManager(project_id=project_id)
    topics_path = paths.allocate_path(ProjectPaths.Topics(column))
    with open(topics_path, 'r', encoding='utf-8') as f:
      return TopicModelingResultModel.model_validate(
        orjson.loads(f.read())
      )
