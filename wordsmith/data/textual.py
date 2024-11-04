from enum import Enum
from typing import Any, Optional, Sequence
import gensim
import pydantic
import spacy

from common.models.enum import ExposedEnum

class TextPreprocessingConfig(pydantic.BaseModel):
  ignore_tokens: Sequence[str] = pydantic.Field(default_factory=lambda: tuple())
  stopwords: Sequence[str] = pydantic.Field(default_factory=lambda: tuple())
  remove_email: bool = True
  remove_url: bool = True
  remove_number: bool = True
  min_word_frequency: int = 5
  max_word_frequency: float = 1 / 2
  max_unique_words: Optional[int] = None
  min_document_length: int = 5
  min_word_length: int = 3

  def load_nlp(self):
    nlp = spacy.load("en_core_web_sm")
    nlp.Defaults.stop_words |= set(self.stopwords)
    tokenizer: Any = nlp.tokenizer
    for token in self.ignore_tokens:
      if token in nlp.Defaults.stop_words:
        nlp.Defaults.stop_words.remove(token)
      tokenizer.add_special_case(token, [{
        "ORTH": token,
        "LEMMA": token,
      }])
  
    return nlp

  def preprocess(self, raw_documents: Sequence[str])->list[list[str]]:
    nlp = self.load_nlp()
    greedy_corpus: Sequence = []
    dictionary = gensim.corpora.Dictionary()
    spacy_docs = nlp.pipe(raw_documents)
    for doc in spacy_docs:
      tokens = []
      for token in doc:
        remove_email = self.remove_email and token.like_email
        remove_number = self.remove_number and token.like_num
        remove_url = self.remove_url and token.like_url
        invalid_token = token.is_stop or token.is_punct or token.is_space
        empty_token = len(token) < self.min_word_length
        if remove_number or remove_email or remove_url or invalid_token or empty_token:
          continue

        tokens.append(token.lemma_.lower())
      greedy_corpus.append(tokens)
    dictionary.add_documents(greedy_corpus)
    dictionary.filter_extremes(
      no_below=self.min_word_frequency,
      no_above=self.max_word_frequency,
      keep_n=10000000000000000 if self.max_unique_words is None else self.max_unique_words,
      keep_tokens=self.ignore_tokens
    )

    corpus: list[list[str]] = []
    for doc in greedy_corpus:
      filtered_doc = list(filter(lambda token: token in dictionary.token2id, doc))
      if len(filtered_doc) < self.min_document_length:
        corpus.append([])
        continue
      corpus.append(filtered_doc)
    return corpus


class DocumentEmbeddingMethodEnum(str, Enum):
  Doc2Vec = "doc2vec"
  SBERT = "sbert"
  TFIDF = "tfidf"

ExposedEnum().register(DocumentEmbeddingMethodEnum)

class TopicModelingConfig(pydantic.BaseModel):
  low_memory: bool = False
  min_topic_size: int = 15
  max_topic_size: float = 1 / 10
  max_topics: Optional[int] = None
  n_gram_range: tuple[int, int] = (1, 2)
  seed_topics: Optional[Sequence[Sequence[str]]] = None
  embedding_method: DocumentEmbeddingMethodEnum = DocumentEmbeddingMethodEnum.Doc2Vec

  no_outliers: bool = False
  represent_outliers: bool = False