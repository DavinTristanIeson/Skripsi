import abc
from dataclasses import dataclass, field
import http

import pydantic
import pandas as pd
import scipy.stats

from modules.logger import ProvisionedLogger
from modules.api import ApiError
from modules.config import SchemaColumn, SchemaColumnTypeEnum

@dataclass
class _StatisticTestValidityModel:
  warnings: list[str] = field(default_factory=lambda: [])
  def merge(self, validity: "_StatisticTestValidityModel"):
    self.warnings.extend(validity.warnings)
    return self
  
logger = ProvisionedLogger().provision("TableComparisonEngine")

class EffectSizeResult(pydantic.BaseModel):
  type: str
  value: float

class SignificanceResult(pydantic.BaseModel):
  type: str
  statistic: float
  p_value: float

@dataclass
class _BaseValidatedComparison(abc.ABC):
  column: SchemaColumn
  groups: list[pd.Series]

  @classmethod
  @abc.abstractmethod
  def get_name(cls)->str:
    ...

  @classmethod
  @abc.abstractmethod
  def get_supported_types(cls)->list[SchemaColumnTypeEnum]:
    ...
  
  @abc.abstractmethod
  def _check_is_valid(self)->_StatisticTestValidityModel:
    ...

  def check_is_valid(self)->_StatisticTestValidityModel:
    # For now we only support comparing two groups
    if len(self.groups) < 2:
      raise ApiError(f"At least two groups has to be provided for a statistic test.", http.HTTPStatus.UNPROCESSABLE_ENTITY)
    supported_types = self.get_supported_types()
    if self.column.type not in supported_types:
      raise ApiError(f"{self.get_name()} can only be used to compare columns of type {', '.join(supported_types)}, but received \"{self.column.type}\" instead.", http.HTTPStatus.UNPROCESSABLE_ENTITY)
    return self._check_is_valid()

class _BaseEffectSize(_BaseValidatedComparison, abc.ABC):
  @abc.abstractmethod
  def effect_size(self)->EffectSizeResult:
    ...

class _BaseStatisticTest(_BaseValidatedComparison, abc.ABC):
  @abc.abstractmethod
  def significance(self)->SignificanceResult:
    ...

__all__ = [
  "SignificanceResult",
  "EffectSizeResult",
]
