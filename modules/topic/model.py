import datetime
from typing import Optional

import pydantic

from modules.config import ProjectPathManager, ProjectPaths


class Topic(pydantic.BaseModel):
  id: int
  words: list[tuple[str, float]]
  label: str
  frequency: int

class TopicHierarchy(pydantic.BaseModel):
  id: int
  words: list[tuple[str, float]]
  label: str
  frequency: int
  children: Optional[list["TopicHierarchy"]] = None

class TopicModelingResult(pydantic.BaseModel):
  project_id: str
  topics: list[Topic]
  hierarchy: TopicHierarchy
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
  def load(project_id: str, column: str)->"TopicModelingResult":
    import orjson

    paths = ProjectPathManager(project_id=project_id)
    topics_path = paths.allocate_path(ProjectPaths.Topics(column))
    with open(topics_path, 'r', encoding='utf-8') as f:
      return TopicModelingResult.model_validate(
        orjson.loads(f.read())
      )

__all__ = [
  "Topic",
  "TopicHierarchy",
  "TopicModelingResult"
]