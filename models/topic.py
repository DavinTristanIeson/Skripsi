from dataclasses import dataclass
from typing import Optional
import pydantic

from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.table.filter_variants import TableFilter
from modules.topic.model import TopicModelingResult

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
  use_cached_umap_vectors: bool

class DocumentTopicAssignmentUpdateSchema(pydantic.BaseModel):
  document_id: int
  topic_id: int

class TopicUpdateSchema(pydantic.BaseModel):
  id: int
  label: str

class RefineTopicsSchema(pydantic.BaseModel):
  topics: list[TopicUpdateSchema]
  document_topics: list[DocumentTopicAssignmentUpdateSchema]

class TopicsOfColumnSchema(pydantic.BaseModel):
  filter: Optional[TableFilter]

# Resource
class DocumentPerTopicResource(pydantic.BaseModel):
  id: int
  original: Optional[str]
  preprocessed: Optional[str]
  topic: Optional[int]

class ColumnTopicModelingResultResource(pydantic.BaseModel):
  column: TextualSchemaColumn
  result: Optional[TopicModelingResult]