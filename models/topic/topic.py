import datetime
from typing import TYPE_CHECKING, Optional
import pydantic

from models.config.paths import ProjectPathManager, ProjectPaths

if TYPE_CHECKING:
  from bertopic import BERTopic

class TopicModel(pydantic.BaseModel):
  id: int
  words: list[tuple[str, float]]
  label: str
  embedding: list[float]
  visualization_embedding: list[float]
  frequency: int

class TopicHierarchyModel(pydantic.BaseModel):
  id: int
  words: list[tuple[str, float]]
  children: Optional[list["TopicHierarchyModel"]] = None

class TopicModelingResultModel(pydantic.BaseModel):
  project_id: str
  topics: list[TopicModel]
  hierarchy: TopicHierarchyModel
  created_at: datetime.datetime = pydantic.Field(
    default_factory=lambda: datetime.datetime.now()
  )

  def save_to_json(self, model: BERTopic, column: str):
    paths = ProjectPathManager(project_id=self.project_id)
    topics_path = paths.allocate_path(ProjectPaths.Topics(column))
    model.save(topics_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=False)

