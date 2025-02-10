from dataclasses import dataclass
from enum import Enum
import functools
import http
import os
from typing import TYPE_CHECKING, Any, Optional, Sequence, cast
import abc

import numpy as np
import numpy.typing as npt

from common.logger import RegisteredLogger, TimeLogger
from common.models.api import ApiError
from common.task.executor import TaskPayload
from controllers.topic.cache import CachedEmbeddingModel
from controllers.topic.utils import BERTopicColumnIntermediateResult
from models.config import Config, TextualSchemaColumn, DocumentEmbeddingMethodEnum
from models.config.paths import ProjectPathManager, ProjectPaths

if TYPE_CHECKING:
  from gensim.models import Doc2Vec
  from sklearn.base import BaseEstimator, TransformerMixin

logger = RegisteredLogger().provision("Topic Modeling")

class BERTopicEmbeddingModelPreprocessingPreference(str, Enum):
  Light = "light"
  Heavy = "heavy"

@dataclass
class BaseBertopicEmbeddingModel(CachedEmbeddingModel, abc.ABC, BaseEstimator, TransformerMixin):
  paths: ProjectPathManager
  column: TextualSchemaColumn

  @classmethod
  @abc.abstractmethod
  def preference(cls)->BERTopicEmbeddingModelPreprocessingPreference:
    ...

  @abc.abstractmethod
  def save(self):
    ...

  @property
  def embedding_path(self):
    return self.paths.full_path(ProjectPaths.DocumentEmbeddings(self.column.name))


@dataclass
class Doc2VecEmbeddingModel(BaseBertopicEmbeddingModel):
  __pre_trained = False

  @classmethod
  def preference(cls):
    return BERTopicEmbeddingModelPreprocessingPreference.Heavy

  @functools.cached_property
  def embedding_model_path(self):
    return self.paths.full_path(ProjectPaths.EmbeddingModel(self.column.name, "doc2vec"))
  
  @functools.cached_property
  def model(self)->Doc2Vec:
    try:
      import gensim
    except ImportError:
      raise ApiError("The gensim library must be installed before SBERT document embedding can be performed.", 400)
  
    if os.path.exists(self.embedding_model_path):
      try:
        model = cast(Doc2Vec, gensim.models.Doc2Vec.load(self.embedding_model_path))
        self.__pre_trained = True
        return model
      except Exception as e:
        logger.error(e)
        logger.error("Failed to load cached Doc2Vec model. Creating a new model...")
    # The documents will have been preprocessed so min_count=1 is fine.
    return gensim.models.Doc2Vec(dm=0, dbow_words=0, min_count=1, vector_size=100)
  
  def save(self):
    os.makedirs(os.path.dirname(self.embedding_model_path), exist_ok=True)
    self.model.save(self.embedding_model_path)

  def fit(self, X: Sequence[str]):
    if self.__pre_trained:
      return self
    
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
    return self
  
  def transform(self, X: Sequence[str]):
    if self.has_cached_embeddings():
      return self.load_cached_embeddings()
    
    with TimeLogger(logger, "Transforming documents to embeddings with Doc2Vec", report_start=True):
      embeddings = np.array(tuple(
        self.model.infer_vector(doc.split())
        for doc in X
      ))
    self.save_embeddings(embeddings)

class SbertEmbeddingModel(BaseBertopicEmbeddingModel):
  @classmethod
  def preference(cls):
    return BERTopicEmbeddingModelPreprocessingPreference.Light
  
  @functools.cached_property
  def model(self):
    try:
      from sentence_transformers import SentenceTransformer
    except ImportError:
      raise ApiError("The sentence_transformers library must be installed before SBERT document embedding can be performed.", 400)
    return SentenceTransformer("all-MiniLM-L6-v2")

  def save(self):
    pass

  def fit(self, X: Sequence[str]):
    return self
  
  def transform(self, X: Sequence[str]):
    if self.has_cached_embeddings():
      return self.load_cached_embeddings()
    # Use the original documents since SBERT performs better with full data
    with TimeLogger(logger, "Transforming documents to embeddings with SBERT", report_start=True):
      embeddings = self.model.encode(X, show_progress_bar=True) # type: ignore
    self.save_embeddings(embeddings)
    return embeddings

@dataclass
class LsaEmbeddingModel(BaseBertopicEmbeddingModel):
  __pre_trained = False

  @classmethod
  def preference(cls):
    return BERTopicEmbeddingModelPreprocessingPreference.Heavy

  @functools.cached_property
  def embedding_model_path(self):
    return self.paths.full_path(ProjectPaths.EmbeddingModel(self.column.name, "lsa.pickle"))
  
  @functools.cached_property
  def model(self):
    try:
      import sklearn.feature_extraction.text
      import sklearn.decomposition
      import sklearn.pipeline
    except ImportError:
      raise ApiError("The scikit-learn library must be installed before LSA-based document embedding can be performed.", 400)
    
    import pickle
    if os.path.exists(self.embedding_model_path):
      try:
        with open(self.embedding_model_path, 'rb') as f:
          cached_model = pickle.load(f)
          self.__pre_trained = True
          return cached_model
      except Exception as e:
        logger.error(e)
        logger.error("Failed to load cached LSA model. Creating a new model...")

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
      min_df=1,
      max_df=1,
      ngram_range=(1,1),
      stop_words=None
    )
    svd = sklearn.decomposition.TruncatedSVD(100)
    pipeline = sklearn.pipeline.make_pipeline(vectorizer, svd)
    return pipeline
  
  def save(self):
    import pickle
    embedding_model_dirpath = os.path.dirname(self.embedding_model_path)
    os.makedirs(embedding_model_dirpath, exist_ok=True)
    with open(self.embedding_model_path, 'wb') as f:
      pickle.dump(self.model, f)
      
  def fit(self, X: Sequence[str]):
    if self.__pre_trained:
      return self
    self.model.fit(X)
    return self
  
  def transform(self, X: Sequence[str]):
    if self.has_cached_embeddings():
      return self.load_cached_embeddings()
    # Use the original documents since SBERT performs better with full data
    with TimeLogger(logger, "Transforming documents to embeddings with LSA", report_start=True):
      embeddings = self.model.transform(X, show_progress_bar=True) # type: ignore
    self.save_embeddings(embeddings)
    return embeddings

@dataclass
class BertopicEmbeddingModelFactory:
  paths: ProjectPathManager
  column: TextualSchemaColumn
  def build(self):
    embedding_method = self.column.topic_modeling.embedding_method
    if embedding_method == DocumentEmbeddingMethodEnum.All_MiniLM_L6_V2:
      return SbertEmbeddingModel(paths=self.paths, column=self.column)
    elif embedding_method == DocumentEmbeddingMethodEnum.Doc2Vec:
      return Doc2VecEmbeddingModel(paths=self.paths, column=self.column)
    elif embedding_method == DocumentEmbeddingMethodEnum.LSA:
      return LsaEmbeddingModel(paths=self.paths, column=self.column)
    else:
      raise ApiError(f"Invalid document embedding method: {self.column.topic_modeling.embedding_method}", http.HTTPStatus.UNPROCESSABLE_ENTITY)
 
def bertopic_embedding(
  intermediate: BERTopicColumnIntermediateResult
):
  paths = intermediate.config.paths
  column = intermediate.column
  task = intermediate.task

  embedding_model = BertopicEmbeddingModelFactory(
    paths=paths,
    column=column,
  ).build()
  intermediate.embedding_model = embedding_model

  embedding_path = paths.full_path(ProjectPaths.DocumentEmbeddings(column.name))
  paths.allocate_path(embedding_path)

  if os.path.exists(embedding_path):
    task.progress(f"Loading cached document embeddings for \"{column.name}\" from \"{embedding_path}\".")
    embeddings = np.load(embedding_path)
    intermediate.embeddings = embeddings
    return
  
  task.progress(
    f"Transforming documents of \"{column.name}\" into document embeddings."
  )

  if embedding_model.preference() == BERTopicEmbeddingModelPreprocessingPreference.Light:
    input_documents = intermediate.embedding_documents
  else:
    input_documents = intermediate.documents

  embeddings = embedding_model.fit_transform(input_documents) # type: ignore
  intermediate.embeddings = embeddings
  embedding_model.save()
  
  task.progress(f"All documents from \"{column.name}\" has been successfully embedded using {column.topic_modeling.embedding_method} and saved in \"{embedding_path}\".")
  task.check_stop()