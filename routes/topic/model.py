from dataclasses import dataclass
from typing import Optional
import pydantic

from modules.config.schema.schema_variants import TextualSchemaColumn
from modules.table.filter_variants import TableFilter
from modules.topic.experiments.model import BERTopicHyperparameterCandidate, BERTopicHyperparameterConstraint
from modules.topic.model import Topic, TopicModelingResult

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
  label: Optional[str] = pydantic.Field(min_length=1)
  tags: Optional[list[str]]
  description: Optional[str]

class RefineTopicsSchema(pydantic.BaseModel):
  topics: list[TopicUpdateSchema]
  document_topics: list[DocumentTopicAssignmentUpdateSchema]

class TopicsOfColumnSchema(pydantic.BaseModel):
  filter: Optional[TableFilter]

class TopicModelExperimentSchema(pydantic.BaseModel):
  n_trials: int = pydantic.Field(ge=5)
  constraint: BERTopicHyperparameterConstraint

# Resource
class DocumentPerTopicResource(pydantic.BaseModel):
  id: int
  original: Optional[str]
  preprocessed: Optional[str]
  topic: Optional[int]

class ColumnTopicModelingResultResource(pydantic.BaseModel):
  column: TextualSchemaColumn
  result: Optional[TopicModelingResult]

class TopicVisualizationResource(pydantic.BaseModel):
  topic: Topic
  x: float
  y: float
  frequency: int

class DocumentVisualizationResource(pydantic.BaseModel):
  document: str
  topic: int
  x: float
  y: float

class DocumentTopicsVisualizationResource(pydantic.BaseModel):
  documents: list[DocumentVisualizationResource]
  topics: list[TopicVisualizationResource]