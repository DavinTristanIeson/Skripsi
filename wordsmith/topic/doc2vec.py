from typing import Sequence, Union, cast
import sklearn.base
import gensim
import numpy as np

from common.logger import RegisteredLogger, TimeLogger
from wordsmith.data.preprocessing import PreprocessingConfig


logger = RegisteredLogger().provision("Doc2Vec")
class Doc2VecTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
  model: gensim.models.Doc2Vec
  preprocessing: PreprocessingConfig
  __built_vocab = False
  def __init__(self, preprocessing: PreprocessingConfig):
    self.model = gensim.models.Doc2Vec()
    self.preprocessing = preprocessing

  def preprocess(self, X: Sequence[Union[str, Sequence[str]]])->Sequence[Sequence[str]]:
    with TimeLogger(logger, "Preprocessing Doc2Vec Input", report_start=True):
      preprocessed_documents: Sequence[Sequence[str]]
      if isinstance(X[0], str):
        preprocessed_documents = tuple(self.preprocessing.preprocess(cast(Sequence[str], X)))
      else:
        preprocessed_documents = X
    return preprocessed_documents

  def fit(self, X: Sequence[Union[str, Sequence[str]]]):
    corpus = self.preprocess(X)
    training_corpus = tuple(
      gensim.models.doc2vec.TaggedDocument(corpus, (idx,))
      for idx, doc in enumerate(X)
    )

    with TimeLogger(logger, "Training Doc2Vec", report_start=True):
      self.model.build_vocab(training_corpus, update=self.__built_vocab)
      self.__built_vocab = True
      self.model.train(training_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)
    return self
  
  def transform(self, X: Sequence[Union[str, Sequence[str]]]):
    corpus = self.preprocess(X)
    with TimeLogger(logger, "Transforming documents to embeddings", report_start=True):
      return np.array(tuple(self.model.infer_vector(doc) for doc in corpus))
