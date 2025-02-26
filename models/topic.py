from typing import Optional
import pandas as pd
import pydantic

from modules.topic.model import Topic, TopicHierarchy, TopicModelingResult

# Schema
class StartTopicModelingSchema(pydantic.BaseModel):
  use_cached_document_embeddings: bool
  use_preprocessed_documents: bool
  use_cached_umap_embeddings: bool
  targets: Optional[list[str]]

class DocumentTopicMappingUpdateSchema(pydantic.BaseModel):
  document_id: int
  topic_id: int

class RefineTopicsSchema(pydantic.BaseModel):
  topics: list[Topic]
  hierarchy: TopicHierarchy
  document_topics: list[DocumentTopicMappingUpdateSchema]

  def apply_update(self, result: TopicModelingResult, data: pd.Series):
    # Update topic modeling result
    result.topics = self.topics
    result.hierarchy = self.hierarchy
    result.reindex()

    # Update document-topic mapping
    document_indices = list(map(lambda x: x.document_id, self.document_topics))
    new_topics = list(map(lambda x: x.topic_id, self.document_topics))
    data[document_indices] = new_topics

# Resource
class DocumentPerTopicResource(pydantic.BaseModel):
  id: int
  original: str
  preprocessed: str
  topic: int