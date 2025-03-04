import datetime
from typing import Optional

import pandas as pd
import pydantic

from modules.baseclass import ValueCarrier
from modules.project.paths import ProjectPathManager, ProjectPaths


class Topic(pydantic.BaseModel):
  id: int
  words: list[tuple[str, float]]
  label: Optional[str]
  frequency: int
  children: Optional[list["Topic"]] = None

  @property
  def is_base(self):
    return self.children is None
  
  def find(self, topic_id: int)->Optional["Topic"]:
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
      child.reindex(new_id)
    self.children = sorted(self.children, key=lambda topic: topic.id)
  
  def iterate_base_topics(self):
    if self.children is None:
      yield self
      return
    for child in self.children:
      yield from child.iterate_base_topics()

  def iterate_all_topics(self):
    bfsq: list[Topic] = [self]
    yield self
    while len(bfsq) > 0:
      current = bfsq.pop(0)
      if not current.children:
        continue
      for child in current.children:
        yield child
        bfsq.append(child)

  def get_topic_scopes(self)->dict[int, list[int]]:
    if self.children is None:
      return {self.id: [self.id]}
    scopes = dict()
    my_scopes:list[int] = []
    for child in self.children:
      child_scope = child.get_topic_scopes()
      scopes[child.id] = child_scope
      my_scopes.extend(child_scope)
    scopes[self.id] = my_scopes
    return scopes
  
  def recalculate_frequency(self):
    if self.children is None:
      return self.frequency
    frequency = 0
    for child in self.children:
      frequency += child.recalculate_frequency()
    self.frequency = frequency
    return frequency
    

class TopicModelingResult(pydantic.BaseModel):
  project_id: str
  topics: Topic
  valid_count: int
  outlier_count: int
  invalid_count: int
  total_count: int
  created_at: datetime.datetime = pydantic.Field(
    default_factory=lambda: datetime.datetime.now()
  )

  @staticmethod
  def infer_from(project_id: str, document_topics: pd.Series, topics: Topic):
    total_count = len(document_topics)
    
    invalid_mask = document_topics.isna()
    invalid_count = invalid_mask.sum()

    outlier_mask = document_topics == -1
    outlier_count = outlier_mask.sum()

    valid_mask = ~(invalid_mask | outlier_mask)
    valid_count = valid_mask.sum()

    return TopicModelingResult(
      invalid_count=invalid_count,
      outlier_count=outlier_count,
      project_id=project_id,
      topics=topics,
      total_count=total_count,
      valid_count=valid_count,
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
    topics_path = paths.assert_path(ProjectPaths.Topics(column))
    with open(topics_path, 'r', encoding='utf-8') as f:
      return TopicModelingResult.model_validate(
        orjson.loads(f.read())
      )
    
  def find(self, topic_id: int)->list[Topic]:
    hierarchy_start = self.topics.find(topic_id)
    if hierarchy_start is None:
      return []
    return list(hierarchy_start.iterate_base_topics())
  
  def reindex(self):
    all_topics = reversed(list(self.topics.iterate_all_topics()))
    new_topic_id = 0
    for topic in all_topics:
      topic.id = new_topic_id
      new_topic_id += 1

__all__ = [
  "Topic",
  "TopicModelingResult"
]