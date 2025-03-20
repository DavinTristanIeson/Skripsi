from dataclasses import dataclass
from typing import Optional
import pydantic

@dataclass
class TopicModelingTaskRequest:
  project_id: str
  column: str

  @property
  def task_id(self):
    return f"{self.project_id}__topic-modeling__{self.column}"

# Schema
class StartTopicModelingSchema(pydantic.BaseModel):
  use_cached_document_vectors: bool
  use_preprocessed_documents: bool
  use_cached_umap_embeddings: bool

class DocumentTopicMappingUpdateSchema(pydantic.BaseModel):
  document_id: int
  topic_id: int

class TopicUpdateSchema(pydantic.BaseModel):
  id: int
  label: Optional[str]
  children: Optional[list["TopicUpdateSchema"]]

class RefineTopicsSchema(pydantic.BaseModel):
  topics: TopicUpdateSchema
  document_topics: list[DocumentTopicMappingUpdateSchema]

# Resource
class DocumentPerTopicResource(pydantic.BaseModel):
  id: int
  original: str
  preprocessed: str
  topic: int