from dataclasses import dataclass
import functools
import http
import os
from typing import TYPE_CHECKING, Any, Optional, Sequence, cast
import abc

import numpy as np
import numpy.typing as npt
import sklearn.pipeline


from common.logger import RegisteredLogger, TimeLogger
from common.models.api import ApiError
from common.task.executor import TaskPayload
from models.config import Config, TextualSchemaColumn, DocumentEmbeddingMethodEnum
from models.config.paths import ProjectPathManager, ProjectPaths

if TYPE_CHECKING:
  from gensim.models import Doc2Vec
  from sklearn.base import BaseEstimator, TransformerMixin

logger = RegisteredLogger().provision("Topic Modeling")

@dataclass
class BaseBertopicEmbeddingModel(abc.ABC, BaseEstimator, TransformerMixin):
  paths: ProjectPathManager
  column: TextualSchemaColumn

  @abc.abstractmethod
  def save(self):
    ...

  @functools.cached_property
  def embedding_directory(self):
    return self.paths.full_path(ProjectPaths.Embeddings)
    
  @functools.cached_property
  def embedding_path(self):
    return os.path.join(embedding_dirpath, self.column.name)

  def save_embeddings(self, embeddings: npt.NDArray):
    embedding_path = self.embedding_path
    os.makedirs(embedding_dirpath, exist_ok=True)
    np.save(embedding_path, embeddings)

  def load_cached_embeddings(self)->Optional[npt.NDArray]:
    embedding_path = self.assert_embedding_path()
    if os.path.exists(embedding_path):
      return np.load(embedding_path)
    return None
    

@dataclass
class Doc2VecEmbeddingModel(BaseBertopicEmbeddingModel):
  config: Config
  column: TextualSchemaColumn
  __built_vocab = False
  __pre_trained = False

  @functools.cached_property
  def embedding_model_path(self):
    embedding_model_dirpath = self.config.paths.full_path(ProjectPaths.EmbeddingModels)
    return os.path.join(embedding_model_dirpath, f"{self.column.name}.doc2vec")

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
    embedding_model_dirpath = os.path.dirname(embedding_model_path)
    os.makedirs(embedding_model_dirpath, exist_ok=True)
    self.model.save(self.embedding_model_path)

  def fit(self, X: Sequence[str]):
    if self.__pre_trained:
      return
    
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
    with TimeLogger(logger, "Transforming documents to embeddings with Doc2Vec", report_start=True):
      return np.array(tuple(
        self.model.infer_vector(doc.split())
        for doc in X
      ))

class SbertTransformerEmbeddingModel(BaseBertopicEmbeddingModel):
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
    # Use the original documents since SBERT performs better with full data
    with TimeLogger(logger, "Transforming documents to embeddings with SBERT", report_start=True):
      embeddings = self.model.encode(X, show_progress_bar=True) # type: ignore
    return embeddings

@dataclass
class LsaTransformerEmbeddingModel(BaseBertopicEmbeddingModel):
  config: Config
  column: TextualSchemaColumn

  @functools.cached_property
  def embedding_model_path(self):
    embedding_model_dirpath = self.config.paths.full_path(ProjectPaths.EmbeddingModels)
    return os.path.join(embedding_model_dirpath, f"{self.column.name}.lsa")
  
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
      with open(self.embedding_model_path, 'rb') as f:
        return pickle.load(f)

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
    return self.model.fit(X)
  
  def transform(self, X: Sequence[str]):
    # Use the original documents since SBERT performs better with full data
    with TimeLogger(logger, "Transforming documents to embeddings with SBERT", report_start=True):
      embeddings = self.model.encode(X, show_progress_bar=True) # type: ignore
    return embeddings

@dataclass
class BertopicEmbeddingModelFactory:
  config: Config
  column: TextualSchemaColumn
  def build(self):
    embedding_method = column.topic_modeling.embedding_method
    if column
    

def bertopic_embedding(
  task: TaskPayload,
  documents: list[str],
  column: TextualSchemaColumn,
  config: Config,
)->npt.NDArray:
  import sklearn.base
  import sklearn.feature_extraction
  import sklearn.decomposition
  import sklearn.pipeline

  embedding_dirpath = config.paths.full_path(ProjectPaths.Embeddings)
  embedding_path = os.path.join(embedding_dirpath, column.name)
  embedding_method = column.topic_modeling.embedding_method

  if os.path.exists(embedding_path):
    task.progress(f"Loading cached document embeddings for \"{column.name}\" from \"{embedding_path}\".")
    return np.load(embedding_path)

  task.progress(f"Embedding all documents from \"{column.name}\" using {embedding_method}...")
  if embedding_method == DocumentEmbeddingMethodEnum.All_MiniLM_L6_V2:
      
    embedding_model = sbert
  elif embedding_method == DocumentEmbeddingMethodEnum.Doc2Vec:
    with TimeLogger(logger, "Fitting doc2vec", report_start=True):
      doc2vec = Doc2VecTransformer()
      doc2vec.fit(documents)
      embeddings = doc2vec.transform(documents)
      embedding_model = doc2vec
  elif embedding_method == DocumentEmbeddingMethodEnum.TFIDF:
    with TimeLogger(logger, "Fitting TF-IDF Vectorizer", report_start=True):
      vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        min_df=1,
        max_df=1,
        ngram_range=(1,1),
        stop_words=None,
      )
      pipeline: list[sklearn.base.BaseEstimator] = [vectorizer]
      sparse_embeddings = vectorizer.fit_transform(documents)
      if sparse_embeddings.shape[1] > 100:
        svd = sklearn.decomposition.TruncatedSVD(100)
        embeddings = svd.fit_transform(sparse_embeddings)
        pipeline.append(svd)
      else:
        embeddings: npt.NDArray = np.asarray(sparse_embeddings.todense()) # type: ignore
    embedding_model = sklearn.pipeline.make_pipeline(*pipeline)
  else:
    raise ApiError(f"Invalid document embedding method: {column.topic_modeling.embedding_method}", http.HTTPStatus.UNPROCESSABLE_ENTITY)
  
  os.makedirs(embedding_dirpath, exist_ok=True)
  np.save(embedding_path, embeddings)
  task.progress(f"All documents from \"{column.name}\" has been successfully embedded using {embedding_method} and saved in \"{embedding_path}\".")

  return embeddings

@dataclass
class CachedUMAP:
  """Helper class to reuse cached UMAP embeddings"""
  reduced_embeddings: npt.NDArray
  
  def fit(self, X):
    return
    
  def transform(self, X):
    if self.__has_cached_reduced_embeddings():
      return self.__get_cached_reduced_embeddings()
    umap_embeddings = self.umap_model.transform(X)
    self.__cache(umap_embeddings)
    return umap_embeddings

  def fit_transform(self, X):
    self.fit(X)
    return self.transform(X)
    

def bertopic_reduce_embeddings(embeddings: npt.NDArray, column: TextualSchemaColumn, config: Config, task: TaskPayload)->npt.NDArray:
  from umap import UMAP
  task.progress(f"Reducing the dimensionality of the embeddings of \"{column.name}\" using UMAP...")
  
  umap_model = UMAP(
    n_neighbors=column.topic_modeling.globality_consideration or column.topic_modeling.min_topic_size,
    min_dist=0.1,
    # BERTopic uses 5 dimensions
    n_components=5,
    metric="euclidean",
    low_memory=column.topic_modeling.low_memory
  )
  umap_embeddings_path = config.paths.full_path(f"{ProjectPaths.UMAPEmbeddings}/{column.name}")
  umap_embeddings = umap_model.fit_transform(embeddings)

  task.progress(f"Saved the UMAP embeddings of \"{column.name}\" to \"{umap_embeddings_path}\"")
  np.save(umap_embeddings_path, umap_embeddings)

  task.progress(f"Reducing the dimensionality of the embeddings of {column.name} using UMAP to 2D for visualization purposes...")
  umap_model = UMAP(
    n_neighbors=column.topic_modeling.globality_consideration or column.topic_modeling.min_topic_size,
    min_dist=0.1,
    n_components=2
  )