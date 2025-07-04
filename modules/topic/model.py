import datetime
from typing import Optional

import pandas as pd
import pydantic

from modules.exceptions.files import CorruptedFileException, FileNotExistsException
from modules.project.paths import ProjectPathManager, ProjectPaths
from modules.storage.atomic import atomic_write

class Topic(pydantic.BaseModel):
  id: int
  words: list[tuple[str, float]]
  label: Optional[str]
  frequency: int
  description: Optional[str] = None
  tags: Optional[list[str]] = None

  @property
  def default_label(self)->str:
    if self.label:
      return self.label
    if len(self.words) > 0:
      return ', '.join(map(lambda x: x[0], self.words[:3]))
    return f"Topic {self.id + 1}"

class TopicModelingResult(pydantic.BaseModel):
  project_id: str
  topics: list[Topic]
  valid_count: int
  outlier_count: int
  invalid_count: int
  total_count: int
  created_at: datetime.datetime = pydantic.Field(
    default_factory=lambda: datetime.datetime.now()
  )

  @staticmethod
  def infer_from(project_id: str, document_topics: pd.Series, topics: list[Topic]):
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

    with atomic_write(topics_path, mode="text") as f:
      f.write(self.model_dump_json(indent=4))

  @staticmethod
  def load(project_id: str, column: str)->"TopicModelingResult":
    paths = ProjectPathManager(project_id=project_id)
    topics_path = paths.assert_path(ProjectPaths.Topics(column))
    FileNotExistsException.verify(topics_path, error=FileNotExistsException.format_message(
      path=topics_path,
      purpose="topic modeling result",
      problem="This may be because the topic modeling algorithm has not been run before for this column.",
      solution="Try running the topic algorithm on this column."
    ))
    with open(topics_path, 'r', encoding='utf-8') as f:
      try:
        return TopicModelingResult.model_validate_json(f.read())
      except pydantic.ValidationError:
        raise CorruptedFileException(
          message=CorruptedFileException.format_message(
            path=topics_path,
            purpose="topic modeling result",
            solution="Try running the topic modeling algorithm again to fix the file."
          )
        )
    
  def find(self, topic_id: int)->Optional[Topic]:
    for topic in self.topics:
      if topic.id == topic_id:
        return topic
    return None
  
  @property
  def renamer(self):
    renamer: dict[int, str] = {}
    unique_labels: dict[str, int] = {}
    for topic in self.topics:
      label = topic.default_label
      if label in unique_labels:
        new_label_suffix = unique_labels.get(label, 1) + 1
        unique_labels[label] = new_label_suffix
        label = f"{label} ({new_label_suffix})"
      else:
        unique_labels[label] = 1
      renamer[topic.id] = label
    return renamer

__all__ = [
  "Topic",
  "TopicModelingResult",
]