from typing import TYPE_CHECKING

from modules.topic.procedure.utils import _BERTopicColumnIntermediateResult

from ..bertopic_ext import SupportedBERTopicEmbeddingModels, BERTopicEmbeddingModelPreprocessingPreference
if TYPE_CHECKING:
  from gensim.models import Doc2Vec

def bertopic_embedding(
  embedding_model: SupportedBERTopicEmbeddingModels,
  intermediate: _BERTopicColumnIntermediateResult
):
  column = intermediate.column
  task = intermediate.task

  embeddings = embedding_model.load_cached_embeddings()
  if embeddings is not None:
    task.log_success(f"Using cached document vectors for \"{column.name}\" from \"{embedding_model.embedding_path}\".")
    intermediate.embeddings = embeddings
    return
  
  task.log_pending(f"Transforming documents of \"{column.name}\" into document vectors...")

  if embedding_model.preference() == BERTopicEmbeddingModelPreprocessingPreference.Light:
    input_documents = intermediate.embedding_documents
  else:
    input_documents = intermediate.documents

  embeddings = embedding_model.fit_transform(input_documents) # type: ignore
  intermediate.embeddings = embeddings
  embedding_model.save()
  
  task.log_success(f"All documents from \"{column.name}\" has been successfully embedded using {column.topic_modeling.embedding_method} and saved in \"{embedding_model.embedding_path}\".")
