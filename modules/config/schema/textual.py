from enum import Enum
from typing import Any, Optional, Sequence
import pandas as pd
import pydantic
import tqdm

from modules.api import ExposedEnum


class DocumentEmbeddingMethodEnum(str, Enum):
  Doc2Vec = "doc2vec"
  All_MiniLM_L6_V2 = "all-MiniLM-L6-v2"
  LSA = "lsa"

ExposedEnum().register(DocumentEmbeddingMethodEnum)

class DocumentPreprocessingMethodEnum(str, Enum):
  English = "en_core_web_sm"

ExposedEnum().register(DocumentPreprocessingMethodEnum)

class TextPreprocessingConfig(pydantic.BaseModel):
  pipeline_type: DocumentPreprocessingMethodEnum = DocumentPreprocessingMethodEnum.English
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

  @property
  def spacy_pipeline_name(self):
    if self.pipeline_type == DocumentPreprocessingMethodEnum.English:
      return "en_core_web_sm"
    raise ValueError(f"Invalid pipeline type: {self.pipeline_type}")

  def load_nlp(self):
    import spacy
    # Tok2vec is needed for POS tagging, POS tagging and attribute ruler is needed for rule-based lemmatization.
    nlp = spacy.load(self.spacy_pipeline_name, disable=["parser", "morphologizer", "ner"])
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

  def preprocess_light(self, raw_documents: Sequence[str])->list[str]:
    nlp = self.load_nlp()
    sbert_corpus: list[str] = []
    spacy_docs = nlp.pipe(raw_documents)
    for doc in tqdm.tqdm(spacy_docs, desc="Preprocessing documents...", total=len(raw_documents)):
      sbert_tokens = []
      for token in doc:
        remove_email = self.remove_email and token.like_email
        remove_url = self.remove_url and token.like_url
        if token.is_space or not token.is_ascii:
          continue
        if remove_email:
          sbert_tokens.append("email")
          continue
        if remove_url:
          sbert_tokens.append("url")
          continue
        sbert_tokens.append(token.text_with_ws)
      sbert_corpus.append(' '.join(sbert_tokens))
    return sbert_corpus

  def preprocess_heavy(self, raw_documents: Sequence[str])->list[str]:
    from gensim.corpora import Dictionary
    
    nlp = self.load_nlp()
    greedy_corpus: list[list[str]] = []
    dictionary = Dictionary()
    
    spacy_docs = nlp.pipe(raw_documents)
    for doc in tqdm.tqdm(spacy_docs, desc="Preprocessing documents...", total=len(raw_documents)):
      tokens = []

      # Topic words preprocessing
      for token in doc:
        remove_email = self.remove_email and token.like_email
        remove_url = self.remove_url and token.like_url
        remove_number = self.remove_number and token.like_num
        invalid_token = token.is_stop or token.is_punct or token.is_space
        empty_token = len(token) < self.min_word_length

        if remove_email:
          tokens.append("EMAIL")
          continue
        if remove_url:
          tokens.append("URL")
          continue
        if remove_number:
          tokens.append("NUMBER")
          continue

        if (invalid_token or empty_token):
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

    corpus: list[str] = []
    for doc in greedy_corpus:
      filtered_doc = list(filter(lambda token: token in dictionary.token2id, doc))
      if len(filtered_doc) < self.min_document_length:
        corpus.append(None) # type: ignore
        continue
      corpus.append(' '.join(filtered_doc))
    return corpus
  

class TopicModelingConfig(pydantic.BaseModel):
  low_memory: bool = False

  # https://stackoverflow.com/questions/67898039/hdbscan-difference-between-parameters

  # Minimal number of topics
  min_topic_size: int = pydantic.Field(default=15, gt=1)
  # Maximum number of topics
  max_topic_size: float = pydantic.Field(default=1 / 5, gt=0.0, le=1.0)

  # Higher value produces more outliers. Lower value might make outliers be included into the topics. This corresponds to HDBSCAN min_samples * min_cluster_size
  clustering_conservativeness: float = pydantic.Field(default=1, gt=0.0, le=1.0)
  
  # Corresponds to UMAP n_neighbors parameter. By default we set this equal to min_topic_size. We keep min_dist=0.1 to help clustering since higher min_dist softens the grouping.
  # This determines the shape of the embedding. Higher values means UMAP will consider the global structure more when reducing the dimensions of the embedding.
  globality_consideration: Optional[int] = pydantic.Field(default=None, gt=1)

  max_topics: Optional[int] = pydantic.Field(default=None, gt=0)
  n_gram_range: tuple[int, int] = (1, 2)
  embedding_method: DocumentEmbeddingMethodEnum = DocumentEmbeddingMethodEnum.All_MiniLM_L6_V2

  # How similar should topics be to be grouped under the same super-topic
  super_topic_similarity: float = pydantic.Field(default=0.7, gt=0.0, le=1.0)
  top_n_words: int = pydantic.Field(default=50, ge=3)

  @pydantic.field_validator("n_gram_range", mode="after")
  def __n_gram_range_validator(cls, value: tuple[int, int]):
    if value[0] > value[1]:
      return (value[1], value[0])
    return value

  no_outliers: bool = False
  represent_outliers: bool = False

__all__ = [
  "TextPreprocessingConfig",
  "TopicModelingConfig",
  "DocumentEmbeddingMethodEnum",
  "DocumentPreprocessingMethodEnum"
]