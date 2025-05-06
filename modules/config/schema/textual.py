from enum import Enum
from typing import Annotated, Any, Optional, Sequence, cast
import pandas as pd
import pydantic
import tqdm

from modules.api import ExposedEnum
from modules.validation.array import StartBeforeEndValidator


class DocumentEmbeddingMethodEnum(str, Enum):
  Doc2Vec = "doc2vec"
  All_MiniLM_L6_V2 = "all-MiniLM-L6-v2"
  LSA = "lsa"

ExposedEnum().register(DocumentEmbeddingMethodEnum)

class DocumentPreprocessingMethodEnum(str, Enum):
  English = "en_core_web_sm"

ExposedEnum().register(DocumentPreprocessingMethodEnum)

class TextPreprocessingConfig(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(use_enum_values=True)

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
  n_gram_range: Annotated[tuple[int, int], StartBeforeEndValidator]

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
        "NORM": token,
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
    nlp = self.load_nlp()
    greedy_corpus: list[str] = []
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
      greedy_corpus.append(' '.join(tokens))

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
      min_df=self.min_df,
      max_df=self.max_df,
      max_features=self.max_unique_words,
      ngram_range=self.n_gram_range, 
    )

    # Train naive vocabulary
    vectorizer.fit(greedy_corpus)
    if len(self.ignore_tokens) > 0:
      vocabulary = list(vectorizer.vocabulary_.keys())
      vocabulary.extend(self.ignore_tokens)
      # No need to set min_df and max_df anymore
      vectorizer = CountVectorizer(vocabulary=vocabulary)

    analyzer = vectorizer.build_analyzer()

    corpus: list[str] = list(map(
      # This is an intentional cast
      lambda doc: cast(str, None if len(doc) < self.min_document_length else ' '.join(doc)),
      map(analyzer, greedy_corpus)
    ))

    return corpus
  

class TopicModelingConfig(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(use_enum_values=True)

  # https://stackoverflow.com/questions/67898039/hdbscan-difference-between-parameters

  # Minimal number of topics
  min_topic_size: int = pydantic.Field(default=15, ge=2)
  # Maximum number of topics
  max_topic_size: Optional[float] = pydantic.Field(default=None, gt=0.0, le=1.0)

  # How many documents with the same theme is required to make us confident that they are part of the same topic.
  topic_confidence_threshold: Optional[int] = pydantic.Field(default=None, ge=2)

  @pydantic.field_validator("topic_confidence_threshold", mode="after")
  def __validate_topic_confidence_threshold(cls, value: int, info: pydantic.ValidationInfo):
    return min(info.data["min_topic_size"], value)

  
  # Corresponds to UMAP n_neighbors parameter. By default we set this equal to min_topic_size. We keep min_dist=0.1 to help clustering since higher min_dist softens the grouping.
  # This determines the shape of the embedding. Higher values means UMAP will consider the global structure more when reducing the dimensions of the embedding.
  reference_document_count: int = pydantic.Field(default=15, ge=2)

  max_topics: Optional[int] = pydantic.Field(default=None, ge=1)

  embedding_method: DocumentEmbeddingMethodEnum = DocumentEmbeddingMethodEnum.All_MiniLM_L6_V2
  
  top_n_words: int = pydantic.Field(default=50, ge=3)

  no_outliers: bool = False
  represent_outliers: bool = False

__all__ = [
  "TextPreprocessingConfig",
  "TopicModelingConfig",
  "DocumentEmbeddingMethodEnum",
  "DocumentPreprocessingMethodEnum"
]