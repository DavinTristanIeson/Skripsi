import abc
from dataclasses import Field, dataclass, field
from enum import Enum
from typing import Callable, ClassVar, Iterable, Literal, Optional, Sequence, cast

import pydantic
import re
import nltk
from nltk.corpus import stopwords
import gensim
import os
import json

from .config import *
from ..common import RelevantDataPaths, SupportedLanguageEnum

RawDocumentBatch = Iterable[Sequence[str]]

def filter_empty_token(batch: Iterable[Optional[str]])->Sequence[str]:
  return tuple(token for token in batch if token is not None and len(token) > 0)

class PreprocessingModuleMode(Enum):
  Online = 'online'
  Final = 'final'

# ========================================================

@dataclass
class BasePreprocessingModule(abc.ABC):
  key: PreprocessingModuleKeys
  context: PreprocessingConfigContext
  mode: ClassVar[PreprocessingModuleMode]

  @abc.abstractmethod
  def apply(self, batch: RawDocumentBatch)->RawDocumentBatch:
    pass

  def after(self)->None:
    pass

  def fit(self, batch: RawDocumentBatch)->None:
    pass
  
  def map_document(self, fn: Callable[[str], Optional[str]], tokens: Sequence[str])->Sequence[str]:
    return filter_empty_token(
      fn(token)
      if token not in self.context.preserve_tokens
      else token
      for token in tokens
    )
  
  def map_batch(self, fn: Callable[[str], Optional[str]], batch: Iterable[Sequence[str]])->Iterable[Sequence[str]]:
    return (self.map_document(fn, doc) for doc in batch)

# ========================================================

@dataclass
class CasefoldPreprocessingModule(BasePreprocessingModule):
  mode = PreprocessingModuleMode.Online
  def process(self, token: str)->str:
    return token.lower()
  def apply(self, batch: RawDocumentBatch)->RawDocumentBatch:
    return self.map_batch(self.process, batch)
  
  @staticmethod
  def parse(config: CasefoldPreprocessingModuleConfig, global_config: PreprocessingConfigContext)->"CasefoldPreprocessingModule":
    return CasefoldPreprocessingModule(key=config.key, context=global_config)

# ========================================================

@dataclass
class RegexFilterPreprocessingModule(BasePreprocessingModule):
  mode = PreprocessingModuleMode.Online
  regex: re.Pattern

  def process(self, token: str)->str:
    return re.sub(self.regex, '', token)
  
  def apply(self, batch: RawDocumentBatch)->RawDocumentBatch:
    return self.map_batch(self.process, batch)
  
  @staticmethod
  def parse(config: RegexFilterPreprocessingModuleConfig, global_config: PreprocessingConfigContext)->"RegexFilterPreprocessingModule":
    return RegexFilterPreprocessingModule(key=config.key, regex=config.regex, context=global_config)
  
# ========================================================

@dataclass
class LengthLimitPreprocessingModule(BasePreprocessingModule):
  mode = PreprocessingModuleMode.Online
  min_length: int
  max_length: int

  def process(self, token: str)->Optional[str]:
    if len(token) < self.min_length or len(token) > self.max_length:
      return None
    return token
  
  def apply(self, batch: RawDocumentBatch)->RawDocumentBatch:
    return self.map_batch(self.process, batch)
  
  @staticmethod
  def parse(config: LengthLimitPreprocessingModuleConfig, global_config: PreprocessingConfigContext)->"LengthLimitPreprocessingModule":
    return LengthLimitPreprocessingModule(key=config.key, min_length=config.min_length, max_length=config.max_length, context=global_config)

# ========================================================

@dataclass
class StemmingPreprocessingModule(BasePreprocessingModule):
  mode = PreprocessingModuleMode.Online
  keep_dictionary: bool = True
  dictionary: dict[str, set[str]] = field(init=False, repr=False)
  stemmer: nltk.StemmerI = field(init=False, repr=False)

  def __post_init__(self):
    self.dictionary = {}
    self.stemmer =  nltk.SnowballStemmer("english")

  def after(self):
    self.context.paths.initialize_paths()
    dictionary_path = os.path.join(self.context.paths.path, RelevantDataPaths.StemmingDictionary)
    dictionary = {k: list(v) for k, v in self.dictionary.items()}
    with open(dictionary_path, 'w', encoding='utf-8') as f:
      json.dump(dictionary, f)

  def process_token(self, token: str)->str:
    stemmed_token: str = cast(str, self.stemmer.stem(token))
    if self.keep_dictionary:
      if token not in self.dictionary:
        self.dictionary[token] = set()
      if token != stemmed_token:
        self.dictionary[token].add(stemmed_token)
    return stemmed_token
  
  
  def apply(self, batch: RawDocumentBatch)->RawDocumentBatch:
    return self.map_batch(self.process_token, batch)
  
  @staticmethod
  def parse(config: StemmingPreprocessingModuleConfig, global_config: PreprocessingConfigContext)->"StemmingPreprocessingModule":
    return StemmingPreprocessingModule(key=config.key, keep_dictionary=config.keep_dictionary, context=global_config)

  
# ========================================================

@dataclass
class LemmatizationPreprocessingModule(BasePreprocessingModule):
  mode = PreprocessingModuleMode.Online
  pos_tagging: bool

  pos_tagger: Optional[nltk.TaggerI] = field(init=False, repr=False)
  lemmatizer: nltk.WordNetLemmatizer = field(init=False, repr=False)

  def __post_init__(self):
    if self.pos_tagging:
      self.pos_tagger = nltk.PerceptronTagger()
    else:
      self.pos_tagger = None
    self.lemmatizer = nltk.WordNetLemmatizer()
  
  @staticmethod
  def treebank_wordset_mapper(treebank_tag: str)->str:
    if treebank_tag.startswith('J'):
      return 'a'
    elif treebank_tag.startswith('V'):
      return 'v'
    elif treebank_tag.startswith('N'):
      return 'n'
    elif treebank_tag.startswith('R'):
      return 's'
    else:
      return 'n'
    
  def pos_map(self, pos_tag: str)->str:
    return LemmatizationPreprocessingModule.treebank_wordset_mapper(pos_tag)
    
  def apply(self, batch: RawDocumentBatch)->RawDocumentBatch:
    if self.pos_tagger is None:
      return self.map_batch(self.lemmatizer.lemmatize, batch)
    
    tagged_batch = cast(Sequence[Sequence[tuple[str, str]]], self.pos_tagger.tag_sents(batch))
    return (
      filter_empty_token(
        self.lemmatizer.lemmatize(token, self.pos_map(pos))
        if token not in self.context.preserve_tokens else token
        for token, pos in tagged_doc
      )
      for tagged_doc in tagged_batch
    )
  
  @staticmethod
  def parse(config: LemmatizationPreprocessingModuleConfig, global_config: PreprocessingConfigContext)->"LemmatizationPreprocessingModule":
    return LemmatizationPreprocessingModule(key=config.key, pos_tagging=config.pos_tagging, context=global_config)
  
# ========================================================

@dataclass
class PhraseDetectionPreprocessingModule(BasePreprocessingModule):
  mode = PreprocessingModuleMode.Final
  phrases: gensim.models.Phrases = field(init=False, repr=False)
  delimiter: str = field(default='_')

  def __post_init__(self):
    connector_words: frozenset[str] = gensim.models.phrases.ENGLISH_CONNECTOR_WORDS if self.context.language == SupportedLanguageEnum.English else frozenset()
    self.phrases = gensim.models.Phrases(delimiter=self.delimiter, connector_words=connector_words)
    return self

  def fit(self, batch: RawDocumentBatch)->None:
    self.phrases.add_vocab(batch)

  def apply(self, batch: RawDocumentBatch)->RawDocumentBatch:
    return (
      cast(Sequence[str], self.phrases[doc])
      for doc in batch
    )
  
  @staticmethod
  def parse(config: PhraseDetectionPreprocessingModuleConfig, global_config: PreprocessingConfigContext)->"PhraseDetectionPreprocessingModule":
    return PhraseDetectionPreprocessingModule(key=config.key, delimiter=config.delimiter, context=global_config)
  
# ========================================================

@dataclass
class TokenLimitPreprocessingModule(BasePreprocessingModule):
  mode = PreprocessingModuleMode.Final
  min_df: int
  max_df: float
  keep_n: int
  dictionary: gensim.corpora.Dictionary = field(init=False, repr=False)
  __has_apply_been_called: bool = field(init=False, repr=False)

  def __post_init__(self):
    self.dictionary = gensim.corpora.Dictionary()
    self.__has_apply_been_called = False

  def fit(self, batch: RawDocumentBatch)->None:
    self.dictionary.add_documents(batch)

  def after(self):
    self.context.paths.initialize_paths()
    dictionary_path = os.path.join(self.context.paths.path, RelevantDataPaths.GensimDictionary)
    with open(dictionary_path, 'wb') as f:
      self.dictionary.save(f)

  def apply(self, batch: RawDocumentBatch)->RawDocumentBatch:
    if not self.__has_apply_been_called:
      self.dictionary.filter_extremes(
        no_below=self.min_df,
        no_above=self.max_df,
        keep_n=self.keep_n,
        keep_tokens=self.context.preserve_tokens
      )
      self.__has_apply_been_called = True

    return (
      tuple(token for token in doc
      if token in self.dictionary.token2id)
      for doc in batch
    )
  
  @staticmethod
  def parse(config: TokenLimitPreprocessingModuleConfig, global_config: PreprocessingConfigContext)->"TokenLimitPreprocessingModule":
    return TokenLimitPreprocessingModule(key=config.key, min_df=config.min_df, max_df=config.max_df, keep_n=config.keep_n, context=global_config)
  
# ========================================================

@dataclass
class StopWordsFilterPreprocessingModule(BasePreprocessingModule):
  mode = PreprocessingModuleMode.Online
  stop_words: set[str]

  @pydantic.model_validator(mode='after')
  def __post_init__(self):
    return self
  
  def process(self, token: str)->Optional[str]:
    if token in self.stop_words:
      return None
    return token

  def apply(self, batch: RawDocumentBatch)->RawDocumentBatch:
    return (
      tuple(token for token in doc
      if token not in self.stop_words)
      for doc in batch
    )
  
  @staticmethod
  def parse(config: StopWordsFilterPreprocessingModuleConfig, global_config: PreprocessingConfigContext)->"StopWordsFilterPreprocessingModule":
    stop_words = set(stopwords.words(global_config.language)) | global_config.preserve_tokens | config.extra
    if config.source is not None:
      with open(config.source.path) as f:
        file_stop_words: set[str] = set(f.read().split(config.source.delimiter))
      stop_words.update(*file_stop_words)

    return StopWordsFilterPreprocessingModule(
      key=config.key,
      stop_words=stop_words,
      context=global_config
    )

# ========================================================

@dataclass
class EmailFilterPreprocessingModule(BasePreprocessingModule):
  mode = PreprocessingModuleMode.Online
  placeholder: Optional[str] = field(default=None)
  regex: re.Pattern = field(init=False, repr=False, default=re.compile(r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""))

  def detect_email(self, token: str)->Optional[str]:
    if re.search(self.regex, token) is None:
      return token
    return self.placeholder

  def apply(self, batch: RawDocumentBatch)->RawDocumentBatch:
    return (
      tuple(token for raw_token in doc
      if (token:=self.detect_email(raw_token)) is not None)
      for doc in batch
    )
  
  @staticmethod
  def parse(config: EmailFilterPreprocessingModuleConfig, global_config: PreprocessingConfigContext)->"EmailFilterPreprocessingModule":
    return EmailFilterPreprocessingModule(key=config.key, placeholder=config.placeholder, context=global_config)

# ========================================================

@dataclass
class URLFilterPreprocessingModule(BasePreprocessingModule):
  mode = PreprocessingModuleMode.Online
  placeholder: Optional[str] = field(default=None)
  regex: re.Pattern = field(init=False, repr=False, default=re.compile(r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""))
  
  def detect_url(self, token: str)->Optional[str]:
    try:
      pydantic.AnyUrl(token)
      return token
    except pydantic.ValidationError:
      return self.placeholder


  def apply(self, batch: RawDocumentBatch)->RawDocumentBatch:
    return (
      tuple(token for token in doc
      if re.search(self.regex, token) is None)
      for doc in batch
    )
  
  @staticmethod
  def parse(config: URLFilterPreprocessingModuleConfig, global_config: PreprocessingConfigContext)->"URLFilterPreprocessingModule":
    return URLFilterPreprocessingModule(key=config.key, placeholder=config.placeholder, context=global_config)

# ========================================================

class PreprocessingModule:
  MAPPER = {
    PreprocessingModuleKeys.Casefold: CasefoldPreprocessingModule,
    PreprocessingModuleKeys.EmailFilter: EmailFilterPreprocessingModule,
    PreprocessingModuleKeys.URLFilter: URLFilterPreprocessingModule,
    PreprocessingModuleKeys.RegexFilter: RegexFilterPreprocessingModule,
    PreprocessingModuleKeys.StopWordsFilter: StopWordsFilterPreprocessingModule,
    PreprocessingModuleKeys.LengthLimit: LengthLimitPreprocessingModule,
    PreprocessingModuleKeys.TokenLimit: TokenLimitPreprocessingModule,
    PreprocessingModuleKeys.Stemming: StemmingPreprocessingModule,
    PreprocessingModuleKeys.Lemmatization: LemmatizationPreprocessingModule,
    PreprocessingModuleKeys.PhraseDetection: PhraseDetectionPreprocessingModule,
  }
  @staticmethod
  def from_config(config: PreprocessingModuleConfigUnion, global_config: PreprocessingConfigContext)->BasePreprocessingModule:
    return PreprocessingModule.MAPPER[config.key].parse(config, global_config) # type: ignore

__all__ = [
  "BasePreprocessingModule",
  "PreprocessingModule",
  "PreprocessingModuleMode",
  
  "CasefoldPreprocessingModule",
  "EmailFilterPreprocessingModule",
  "URLFilterPreprocessingModule",
  "RegexFilterPreprocessingModule",
  "StopWordsFilterPreprocessingModule",
  "LengthLimitPreprocessingModule",
  "TokenLimitPreprocessingModule",
  "StemmingPreprocessingModule",
  "LemmatizationPreprocessingModule",
  "PhraseDetectionPreprocessingModule",

]