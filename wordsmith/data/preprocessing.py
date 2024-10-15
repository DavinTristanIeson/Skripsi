import abc
from enum import Enum
from typing import Any, Iterable, Literal, Sequence, Union, cast
import re

import pydantic
import tqdm
import nltk

from .modules import *
from .common import *

class TokenizationTypeEnum(str, Enum):
  NLTK = "nltk",
  Whitespace = "whitespace",
  Regex = "regex"

class BaseTokenizationConfig(abc.ABC):
  @abc.abstractmethod
  def tokenize(self, data: Iterable[str])->Iterable[Sequence[str]]:
    pass

class NLTKTokenizationConfig(pydantic.BaseModel, BaseTokenizationConfig):
  type: Literal[TokenizationTypeEnum.NLTK]
  def tokenize(self, data: Iterable[str])->Iterable[Sequence[str]]:
    return (nltk.word_tokenize(doc, preserve_line=False) for doc in data)
  
class WhitespaceTokenizationConfig(pydantic.BaseModel, BaseTokenizationConfig):
  type: Literal[TokenizationTypeEnum.Whitespace]
  def tokenize(self, data: Iterable[str])->Iterable[Sequence[str]]:
    return (doc.split() for doc in data)
class RegexTokenizationConfig(pydantic.BaseModel, BaseTokenizationConfig):
  type: Literal[TokenizationTypeEnum.Regex]
  regex: re.Pattern

  @pydantic.field_validator("regex", mode="before")
  @classmethod
  def regex_validator(cls, value):
    return re.compile(value)

  def tokenize(self, data: Iterable[str])->Iterable[Sequence[str]]:
    return (re.split(self.regex, doc) for doc in data)

TokenizationConfig = Union[NLTKTokenizationConfig, WhitespaceTokenizationConfig, RegexTokenizationConfig]

class PreprocessingConfig(pydantic.BaseModel):
  tokenization: TokenizationConfig = pydantic.Field(discriminator="type")
  preserve_tokens: set[str] = pydantic.Field(default_factory=lambda: set())
  steps: Sequence[BasePreprocessingModule]
  language: SupportedLanguageEnum = pydantic.Field(default=SupportedLanguageEnum.English)

  token_delimiter: str = pydantic.Field(default=' ')

  @pydantic.model_validator(mode="before")
  def validate_steps(self, info: pydantic.ValidationInfo):
    data = cast(dict[str, Any], self)
    if "steps" not in data:
      raise ValueError("`steps` should be provided")
    language = data.get("language", SupportedLanguageEnum.English)
    preserve_tokens = data.get("preserve_tokens", set())
    pydantic_context = cast(dict, info.context)
    if pydantic_context is None or pydantic_context.get('paths', None) is None:
      raise ValueError("PreprocessingConfig is initialized with a special context. Call Config.parse rather than Config.model_validate.")
    global_config = PreprocessingConfigContext(
      language=language,
      preserve_tokens=preserve_tokens,
      paths=pydantic_context["paths"],
    )
    
    steps = cast(Sequence[Any], data.get("steps"))
    if not hasattr(steps, '__iter__'):
      raise ValueError("Cannot iterate through `steps` field.")
    
    data["steps"] = list(
      PreprocessingModule.from_config(
        PreprocessingModuleConfig.model_validate(
          step
          if not isinstance(step, str)
          else {"key": step}
        ).root,
        global_config
      ) for step in steps
    )

    return self
    
  def apply(self, data: Iterable[str], *, show_progress:bool = True)->Iterable[str]:
    documents = tuple(self.tokenization.tokenize(tqdm.tqdm(data, desc="Tokenization", disable=not show_progress)))

    for step in self.steps:
      if step.__class__.mode == PreprocessingModuleMode.Online:
        documents = tuple(step.apply(tqdm.tqdm(documents, desc=str(step.key), disable=not show_progress)))
      elif step.mode == PreprocessingModuleMode.Final:
        step.fit(documents)
    
    for step in self.steps:
      if step.__class__.mode == PreprocessingModuleMode.Final:
        documents = tuple(step.apply(tqdm.tqdm(documents, desc=str(step.key), disable=not show_progress)))

    for step in self.steps:
      step.after()
    return (self.token_delimiter.join(doc).strip() for doc in documents)
  
__all__ = [
  "PreprocessingConfig",
  "TokenizationTypeEnum"
]