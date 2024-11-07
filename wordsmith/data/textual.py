from enum import Enum
from typing import Any, Optional, Sequence
import gensim
import pydantic
import spacy
import tqdm

from common.models.enum import ExposedEnum

class TextPreprocessingConfig(pydantic.BaseModel):
  ignore_tokens: Sequence[str] = pydantic.Field(default_factory=lambda: tuple())
  stopwords: Sequence[str] = pydantic.Field(default_factory=lambda: tuple())
  remove_email: bool = True
  remove_url: bool = True
  remove_number: bool = True
  min_df: int = pydantic.Field(default=5, gt=0)
  max_df: float = pydantic.Field(default=1/2, gt=0.0, le=1.0)
  max_unique_words: Optional[int] = pydantic.Field(default=None, gt=0)
  min_document_length: int = pydantic.Field(default=5, gt=0)
  min_word_length: int = pydantic.Field(default=3, gt=0)

  def load_nlp(self):
    # Tok2vec is needed for POS tagging, POS tagging and attribute ruler is needed for rule-based lemmatization. NER for detecting named entities.
    nlp = spacy.load("en_core_web_sm", disable=["parser", "morphologizer"])
    nlp.Defaults.stop_words |= set(map(lambda x: x.lower(), self.stopwords))
    tokenizer: Any = nlp.tokenizer
    for token in self.ignore_tokens:
      if token in nlp.Defaults.stop_words:
        nlp.Defaults.stop_words.remove(token.lower())
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
    for doc in tqdm.tqdm(spacy_docs, desc="Preprocessing documents...", total=len(raw_documents)):
      tokens = []
      for token in doc:
        remove_email = self.remove_email and token.like_email
        remove_number = self.remove_number and token.like_num
        remove_url = self.remove_url and token.like_url
        invalid_token = token.is_stop or token.is_punct or token.is_space
        empty_token = len(token) < self.min_word_length

        if (remove_number or remove_email or remove_url or invalid_token or empty_token):
          continue

        tokens.append(token.lemma_.lower())
      greedy_corpus.append(tokens)
    dictionary.add_documents(greedy_corpus)

    dictionary.filter_extremes(
      no_below=self.min_df,
      no_above=self.max_df,
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
  min_topic_size: int = pydantic.Field(default=15, gt=1)
  max_topic_size: float = pydantic.Field(default=1 / 5, gt=0.0, le=1.0)
  max_topics: Optional[int] = pydantic.Field(default=None, gt=0)
  n_gram_range: tuple[int, int] = (1, 2)
  seed_topics: Optional[Sequence[Sequence[str]]] = None
  embedding_method: DocumentEmbeddingMethodEnum = DocumentEmbeddingMethodEnum.Doc2Vec

  @pydantic.field_validator("n_gram_range", mode="after")
  def __n_gram_range_validator(cls, value: tuple[int, int]):
    if value[0] > value[1]:
      return (value[1], value[0])
    return value

  no_outliers: bool = False
  represent_outliers: bool = False