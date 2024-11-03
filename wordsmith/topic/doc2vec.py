from typing import Sequence, cast
import sklearn.base
import gensim
import numpy as np

from common.logger import RegisteredLogger, TimeLogger


logger = RegisteredLogger().provision("BERTopic Components")
class Doc2VecTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
  model: gensim.models.Doc2Vec
  documents_length: int
  __built_vocab = False
  def __init__(self):
    # The documents will have been preprocessed so min_count=1 is fine.
    self.model = gensim.models.Doc2Vec(dm=0, dbow_words=1, min_count=1, vector_size=300)

  def save(self, path: str):
    self.model = cast(gensim.models.Doc2Vec, gensim.models.Doc2Vec.load(path))
    self.documents_length = cast(int, self.model.corpus_count)

  def fit(self, X: Sequence[str]):
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
    with TimeLogger(logger, "Transforming documents to embeddings", report_start=True):
      return np.array(tuple(
        self.model.infer_vector(doc.split())
        for doc in X
      ))
