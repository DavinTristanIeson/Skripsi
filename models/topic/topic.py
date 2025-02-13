import datetime
from typing import TYPE_CHECKING, Optional

import pydantic

from models.config.paths import ProjectPathManager, ProjectPaths


class TopicModel(pydantic.BaseModel):
  id: int
  words: list[tuple[str, float]]
  label: str
  visualization_embedding: list[float]
  frequency: int

class TopicHierarchyModel(pydantic.BaseModel):
  id: int
  words: list[tuple[str, float]]
  frequency: int
  children: Optional[list["TopicHierarchyModel"]] = None

class TopicModelingResultModel(pydantic.BaseModel):
  project_id: str
  topics: list[TopicModel]
  hierarchy: TopicHierarchyModel
  created_at: datetime.datetime = pydantic.Field(
    default_factory=lambda: datetime.datetime.now()
  )

  def save_as_json(self, column: str):
    paths = ProjectPathManager(project_id=self.project_id)
    topics_path = paths.allocate_path(ProjectPaths.Topics(column))
    with open(topics_path, 'w', encoding='utf-8') as f:
      f.write(self.model_dump_json())
