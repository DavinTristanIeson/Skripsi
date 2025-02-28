from typing import Optional
import pandas as pd
import pydantic

from modules.topic.model import Topic, TopicModelingResult

# Schema
class StartTopicModelingSchema(pydantic.BaseModel):
  use_cached_document_embeddings: bool
  use_preprocessed_documents: bool
  use_cached_umap_embeddings: bool
  targets: Optional[list[str]]

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