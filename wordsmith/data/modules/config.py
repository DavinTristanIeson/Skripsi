from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union

import pydantic
import re

from ..cache import ConfigPathsManager
from ..common import *

class PreprocessingModuleKeys(str, Enum):
  Casefold = "casefold"
  RegexFilter = "regex_filter"
  URLFilter = "url_filter"
  EmailFilter = "email_filter"
  StopWordsFilter = "stop_words_filter"
  LengthLimit = "length_limit"
  TokenLimit = "token_limit"
  PhraseDetection = "phrase_detection"
  Stemming = "stemming"
  Lemmatization = "lemmatization"

@dataclass
class PreprocessingConfigContext:
  language: SupportedLanguageEnum
  preserve_tokens: set[str]
  paths: ConfigPathsManager

# ========================================================

class CasefoldPreprocessingModuleConfig(pydantic.BaseModel):
  key: Literal[PreprocessingModuleKeys.Casefold]

# ========================================================

class RegexFilterPreprocessingModuleConfig(pydantic.BaseModel):
  key: Literal[PreprocessingModuleKeys.RegexFilter]
  regex: re.Pattern

  @pydantic.field_validator("regex", mode="before")
  @classmethod
  def regex_validator(cls, value):
    return re.compile(value)
  
# ========================================================

class LengthLimitPreprocessingModuleConfig(pydantic.BaseModel):
  key: Literal[PreprocessingModuleKeys.LengthLimit]
  min_length: int = pydantic.Field(validation_alias="min", default=3)
  max_length: int = pydantic.Field(validation_alias="max", default=20)

# ========================================================

class StemmingPreprocessingModuleConfig(pydantic.BaseModel):
  key: Literal[PreprocessingModuleKeys.Stemming]
  keep_dictionary: bool = True
  
# ========================================================

class LemmatizationPreprocessingModuleConfig(pydantic.BaseModel):
  key: Literal[PreprocessingModuleKeys.Lemmatization]
  pos_tagging: bool = pydantic.Field(default=True)
  
# ========================================================

class PhraseDetectionPreprocessingModuleConfig(pydantic.BaseModel):
  key: Literal[PreprocessingModuleKeys.PhraseDetection]
  delimiter: str = pydantic.Field(default='_')

# ========================================================

class TokenLimitPreprocessingModuleConfig(pydantic.BaseModel):
  key: Literal[PreprocessingModuleKeys.TokenLimit]
  min_df: int = pydantic.Field(default=5)
  max_df: float = pydantic.Field(default=0.5)
  keep_n: int = pydantic.Field(default=200000)

# ========================================================

class StopWordsFilterPreprocessingModuleConfig(pydantic.BaseModel):
  class StopWordsSourceFile(pydantic.BaseModel):
    path: str
    delimiter: str
  key: Literal[PreprocessingModuleKeys.StopWordsFilter]
  extra: set[str] = pydantic.Field(default_factory=lambda: set())
  source: Optional[StopWordsSourceFile] = None

  
# ========================================================

class EmailFilterPreprocessingModuleConfig(pydantic.BaseModel):
  key: Literal[PreprocessingModuleKeys.EmailFilter]
  placeholder: Optional[str] = pydantic.Field(default=None)

# ========================================================

class URLFilterPreprocessingModuleConfig(pydantic.BaseModel):
  key: Literal[PreprocessingModuleKeys.URLFilter]
  placeholder: Optional[str] = pydantic.Field(default=None)

# ========================================================

PreprocessingModuleConfigUnion = Union[
  CasefoldPreprocessingModuleConfig,
  RegexFilterPreprocessingModuleConfig,
  EmailFilterPreprocessingModuleConfig,
  URLFilterPreprocessingModuleConfig,
  StopWordsFilterPreprocessingModuleConfig,
  TokenLimitPreprocessingModuleConfig,
  LengthLimitPreprocessingModuleConfig,
  StemmingPreprocessingModuleConfig,
  LemmatizationPreprocessingModuleConfig,
  PhraseDetectionPreprocessingModuleConfig,
]
class PreprocessingModuleConfig(pydantic.RootModel):
  root: PreprocessingModuleConfigUnion = pydantic.Field(discriminator="key")


__all__ = [
  "PreprocessingModuleConfig",
  "CasefoldPreprocessingModuleConfig",
  "EmailFilterPreprocessingModuleConfig",
  "URLFilterPreprocessingModuleConfig",
  "RegexFilterPreprocessingModuleConfig",
  "StopWordsFilterPreprocessingModuleConfig",
  "LengthLimitPreprocessingModuleConfig",
  "TokenLimitPreprocessingModuleConfig",
  "StemmingPreprocessingModuleConfig",
  "LemmatizationPreprocessingModuleConfig",
  "PhraseDetectionPreprocessingModuleConfig",
  "PreprocessingModuleConfigUnion",

  "PreprocessingModuleKeys",
  "PreprocessingConfigContext"
]