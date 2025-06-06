
import abc
from dataclasses import dataclass, field
from enum import Enum
import functools
import http
import os
from typing import TYPE_CHECKING, Sequence, Union, cast

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from modules.config import TextualSchemaColumn
from modules.exceptions.dependencies import DependencyImportException, InvalidValueTypeException
from modules.logger.provisioner import ProvisionedLogger
from modules.project.paths import ProjectPathManager, ProjectPaths
from modules.config.schema.textual import DocumentEmbeddingMethodEnum
from modules.logger import TimeLogger
from modules.storage.atomic import atomic_write

if TYPE_CHECKING:
  from gensim.models import Doc2Vec

# BASE MODELS
logger = ProvisionedLogger().provision("Topic Modeling")

class BERTopicEmbeddingModelPreprocessingPreference(str, Enum):
  Light = "light"
  Heavy = "heavy"

@dataclass
class _DocumentEmbeddingModelDependency:
  project_id: str
  column: TextualSchemaColumn

@dataclass
class _BaseDocumentEmbeddingModel(_DocumentEmbeddingModelDependency, abc.ABC):
  @classmethod
  @abc.abstractmethod
  def preference(cls)->BERTopicEmbeddingModelPreprocessingPreference:
    ...

@dataclass
class Doc2VecEmbeddingModel(_BaseDocumentEmbeddingModel, BaseEstimator, TransformerMixin):
  def save_model(self, model: "Doc2Vec"):
    embedding_model_path = ProjectPathManager(project_id=self.project_id).full_path(ProjectPaths.EmbeddingModel(self.column.name, "doc2vec"))
    model.save(embedding_model_path)
  
  @functools.cached_property
  def model(self)->"Doc2Vec":
    try:
      from gensim.models import Doc2Vec
      return Doc2Vec(dm=0, dbow_words=0, min_count=1, vector_size=100)
    except ImportError:
      raise DependencyImportException(
        name="gensim",
        purpose="Doc2Vec document embedding can be performed"
      )

  @classmethod
  def preference(cls):
    return BERTopicEmbeddingModelPreprocessingPreference.Heavy

  def fit(self, X: Sequence[str]):
    import gensim
    training_corpus = tuple(
      gensim.models.doc2vec.TaggedDocument(doc.split(), (idx,))
      for idx, doc in enumerate(X)
    )
    self.documents_length = len(X)

    with TimeLogger(logger, "Training Doc2Vec", report_start=True):
      self.model.build_vocab(training_corpus, update=self.__built_vocab)
      self.__built_vocab = True
      self.model.train(training_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)
  
  def transform(self, X: Sequence[str]):
    with TimeLogger(logger, "Transforming documents to embeddings with Doc2Vec", report_start=True):
      return np.array(tuple(
        self.model.infer_vector(doc.split())
        for doc in X
      ))

class SbertEmbeddingModel(_BaseDocumentEmbeddingModel, BaseEstimator, TransformerMixin):
  @classmethod
  def preference(cls):
    return BERTopicEmbeddingModelPreprocessingPreference.Light
  
  @functools.cached_property
  def model(self):
    try:
      from sentence_transformers import SentenceTransformer
    except ImportError:
      raise DependencyImportException(
        name="sentence_transformers",
        purpose="SBERT document embedding can be performed"
      )
    return SentenceTransformer("all-MiniLM-L6-v2")

  def fit(self, X: Sequence[str]):
    # There's no fitting
    return self
  
  def transform(self, X: Sequence[str]):
    # Use the original documents since SBERT performs better with full data
    with TimeLogger(logger, "Transforming documents to embeddings with SBERT", report_start=True):
      embeddings = self.model.encode(X, show_progress_bar=True) # type: ignore
    return embeddings

@dataclass
class LsaEmbeddingModel(_BaseDocumentEmbeddingModel):
  def save_model(self, model: Pipeline):
    import pickle
    embedding_model_path = ProjectPathManager(project_id=self.project_id).full_path(ProjectPaths.EmbeddingModel(self.column.name, "lsa.pickle"))
    embedding_model_dirpath = os.path.dirname(embedding_model_path)
    os.makedirs(embedding_model_dirpath, exist_ok=True)
    with atomic_write(embedding_model_path, mode="binary") as f:
      pickle.dump(model, f)
  
  @functools.cached_property
  def model(self):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import make_pipeline
    vectorizer = TfidfVectorizer(
      min_df=1,
      max_df=1,
      ngram_range=(1,1),
      stop_words=None
    )
    svd = TruncatedSVD(100)
    pipeline = make_pipeline(vectorizer, svd)
    return pipeline


  @classmethod
  def preference(cls):
    return BERTopicEmbeddingModelPreprocessingPreference.Heavy

  def fit(self, X: Sequence[str]):
    self.model.fit(X)
  
  def transform(self, X: Sequence[str]):
    with TimeLogger(logger, "Transforming documents to embeddings with LSA", report_start=True):
      return self.model.transform(X, show_progress_bar=True) # type: ignore

@dataclass
class BERTopicEmbeddingModelFactory:
  project_id: str
  column: TextualSchemaColumn
  def build(self):
    embedding_method = self.column.topic_modeling.embedding_method
    if embedding_method == DocumentEmbeddingMethodEnum.All_MiniLM_L6_V2:
      return SbertEmbeddingModel(project_id=self.project_id, column=self.column)
    elif embedding_method == DocumentEmbeddingMethodEnum.Doc2Vec:
      return Doc2VecEmbeddingModel(project_id=self.project_id, column=self.column)
    elif embedding_method == DocumentEmbeddingMethodEnum.LSA:
      return LsaEmbeddingModel(project_id=self.project_id, column=self.column)
    else:
      raise InvalidValueTypeException(value=self.column.topic_modeling.embedding_method, type="document embedding method")
 
def get_embedding_model_preference(column: TextualSchemaColumn)->BERTopicEmbeddingModelPreprocessingPreference:
  return BERTopicEmbeddingModelFactory(
    # This is not relevant
    project_id=None, # type: ignore
    column=column,
  ).build().preference()

SupportedBERTopicEmbeddingModels = Union[SbertEmbeddingModel, Doc2VecEmbeddingModel, LsaEmbeddingModel]

__all__ = [
  "BERTopicEmbeddingModelFactory",
  "SupportedBERTopicEmbeddingModels",
  "BERTopicEmbeddingModelPreprocessingPreference",
  "get_embedding_model_preference",
]