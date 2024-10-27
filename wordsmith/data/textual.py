from functools import lru_cache
from typing import Any, Iterable, Optional, Sequence
import pydantic
import spacy
from spacy.symbols import ORTH
import spacy.symbols

class TextPreprocessingConfig(pydantic.BaseModel):
  ignore_tokens: Sequence[str] = pydantic.Field(default_factory=lambda: tuple())
  stopwords: Sequence[str] = pydantic.Field(default_factory=lambda: tuple())
  remove_email: bool = True
  remove_url: bool = True
  remove_number: bool = True

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

  def preprocess(self, raw_documents: Iterable[str])->Iterable[list[str]]:
    nlp = self.load_nlp()
    for rawdoc in raw_documents:
      doc = nlp(rawdoc)
      tokens = []
      for token in doc:
        remove_email = self.remove_email and token.like_email
        remove_number = self.remove_number and token.like_num
        remove_url = self.remove_url and token.like_url
        invalid_token = token.is_stop or token.is_punct or token.is_space
        if remove_number or remove_email or remove_url or invalid_token:
          continue

        tokens.append(token.lemma_.lower())
      yield tokens

class TopicModelingConfig(pydantic.BaseModel):
  low_memory: bool = False
  min_topic_size: int = 15
  max_topic_size: float = 1 / 15
  max_topics: Optional[int] = None
  n_gram_range: tuple[int, int] = (1, 2)
  seed_topics: Optional[Sequence[Sequence[str]]] = None

  no_outliers: bool = False
  represent_outliers: bool = False