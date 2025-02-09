import datetime
from typing import Optional
import pydantic


class TopicModel(pydantic.BaseModel):
  id: int
  words: list[tuple[str, float]]
  label: str

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

