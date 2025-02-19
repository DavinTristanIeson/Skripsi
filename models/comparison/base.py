import abc
from dataclasses import dataclass, field
from enum import Enum
import http
from typing import Any, Optional, cast

import pydantic
import pandas as pd
import scipy.stats

from common.logger import RegisteredLogger
from common.models.api import ApiError
from common.models.validators import CommonModelConfig
from models.config.schema import CategoricalSchemaColumn, SchemaColumn, SchemaColumnTypeEnum
from models.table.filter_variants import TableFilter

@dataclass
class StatisticTestValidityModel:
  warnings: list[str] = field(default_factory=lambda: [])
  def merge(self, validity: "StatisticTestValidityModel"):
    self.warnings.extend(validity.warnings)
    return self
  
class NamedTableFilter(pydantic.BaseModel):
  name: str
  filter: TableFilter

logger = RegisteredLogger().provision("TableComparisonEngine")

class EffectSizeResult(pydantic.BaseModel):
  model_config = CommonModelConfig
  type: str
  value: float

class SignificanceResult(pydantic.BaseModel):
  model_config = CommonModelConfig
  type: str
  statistic: float
  p_value: float

@dataclass
class BaseValidatedComparison(abc.ABC):
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

  def check_normality(self, data: pd.Series)->list[str]:
    normaltest_result = scipy.stats.normaltest(data).pvalue
    is_normal = normaltest_result < 0.05
    warnings = []
    if not is_normal:
      warnings.append(f"{self.get_name()} expects the samples to be normally distributed, but \"{data.name}\" does not follow a normal distribution (confidence: {1 - normaltest_result}).")
    return warnings

  @abc.abstractmethod
  def _check_is_valid(self)->StatisticTestValidityModel:
    ...

  def check_is_valid(self)->StatisticTestValidityModel:
    # For now we only support comparing two groups
    if len(self.groups) != 2:
      raise ApiError(f"{self.get_name()} can only be used to compare two groups.", http.HTTPStatus.UNPROCESSABLE_ENTITY)
    supported_types = self.get_supported_types()
    if self.column.type not in supported_types:
      raise ApiError(f"{self.get_name()} can only be used to compare columns of type {', '.join(supported_types)}, but received \"{self.column.type}\" instead.", http.HTTPStatus.UNPROCESSABLE_ENTITY)
    return self._check_is_valid()

class BaseEffectSize(BaseValidatedComparison, abc.ABC):
  @abc.abstractmethod
  def effect_size(self)->EffectSizeResult:
    ...

class BaseStatisticTest(BaseValidatedComparison, abc.ABC):
  @abc.abstractmethod
  def significance(self)->SignificanceResult:
    ...
