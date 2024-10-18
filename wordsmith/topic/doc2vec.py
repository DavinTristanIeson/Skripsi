from typing import Sequence
import sklearn.base
import gensim
import numpy as np

from common.logger import RegisteredLogger, TimeLogger


logger = RegisteredLogger().provision("BERTopic Components")
class Doc2VecTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
  model: gensim.models.Doc2Vec
  __built_vocab = False
  def __init__(self):
    self.model = gensim.models.Doc2Vec()

  def fit(self, X: Sequence[str]):
    training_corpus = tuple(
      gensim.models.doc2vec.TaggedDocument(doc.split(), (idx,))
      for idx, doc in enumerate(X)
    )

    with TimeLogger(logger, "Training Doc2Vec", report_start=True):
      self.model.build_vocab(training_corpus, update=self.__built_vocab)
      self.__built_vocab = True
      self.model.train(training_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)
    return self
  
  def transform(self, X: Sequence[str]):
    with TimeLogger(logger, "Transforming documents to embeddings", report_start=True):
      return np.array(tuple(
        self.model.infer_vector(doc.split())
        for doc in X
      ))
