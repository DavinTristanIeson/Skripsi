import datetime
from typing import Optional

import pydantic

from modules.baseclass import ValueCarrier
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

  def as_topic(self)->Topic:
    return Topic(
      id=self.id,
      words=self.words,
      label=self.label,
      frequency=self.frequency,
    )
  
  def find(self, topic_id: int)->Optional["TopicHierarchy"]:
    if self.id == topic_id:
      return self
    
    if self.children is None:
      return None
    for child in self.children:
      result = child.find(topic_id)
      if result is not None:
        return result
    return None
  
  def reindex(self, new_id: ValueCarrier[int]):
    if self.children is None:
      self.id = new_id.value
      new_id.value += 1
      return
    for child in self.children:
      self.reindex(new_id)
    self.children = sorted(self.children, key=lambda topic: topic.id)
  
  def iterate_topics(self):
    if self.children is None:
      yield self.as_topic()
      return
    topics: list[Topic] = []
    for child in self.children:
      yield from child.iterate_topics()
    return topics    

class TopicModelingResult(pydantic.BaseModel):
  project_id: str
  topics: list[Topic]
  hierarchy: TopicHierarchy
  frequency: int
  created_at: datetime.datetime = pydantic.Field(
    default_factory=lambda: datetime.datetime.now()
  )

  def iterate_topics(self):
    for topic in self.topics:
      yield topic
    yield from self.hierarchy.iterate_topics()

  def save_as_json(self, column: str):
    paths = ProjectPathManager(project_id=self.project_id)
    topics_path = paths.allocate_path(ProjectPaths.Topics(column))
    with open(topics_path, 'w', encoding='utf-8') as f:
      f.write(self.model_dump_json(indent=4))

  @staticmethod
  def load(project_id: str, column: str)->"TopicModelingResult":
    import orjson

    paths = ProjectPathManager(project_id=project_id)
    topics_path = paths.assert_path(ProjectPaths.Topics(column))
    with open(topics_path, 'r', encoding='utf-8') as f:
      return TopicModelingResult.model_validate(
        orjson.loads(f.read())
      )
    
  def find(self, topic_id: int)->list[Topic]:
    for topic in self.topics:
      if topic.id == topic_id:
        return [topic]
    hierarchy_base = self.hierarchy.find(topic_id)
    if hierarchy_base is None:
      return []
    return list(hierarchy_base.iterate_topics())

  def reindex(self):
    topics = sorted(self.topics, key=lambda topic: topic.id)
    new_topic_id = 0
    for topic in topics:
      topic.id = new_topic_id
      new_topic_id += 1
    self.hierarchy.reindex(ValueCarrier(new_topic_id))

__all__ = [
  "Topic",
  "TopicHierarchy",
  "TopicModelingResult"
]