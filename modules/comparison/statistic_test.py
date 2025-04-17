from dataclasses import dataclass
from enum import Enum
import itertools
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats

from modules.comparison.utils import _chisq_prepare, _mann_whitney_u_prepare
from modules.logger import ProvisionedLogger
from modules.api import ExposedEnum
from modules.config import SchemaColumn, SchemaColumnTypeEnum
from .base import _BaseStatisticTest, SignificanceResult, _StatisticTestValidityModel

logger = ProvisionedLogger().provision("TableComparisonEngine")

class StatisticTestMethodEnum(str, Enum):
  T = "t"
  MannWhitneyU = "mann-whitney-u"
  ChiSquared = "chi-squared"

ExposedEnum().register(StatisticTestMethodEnum)

class TStatisticTest(_BaseStatisticTest):
  @classmethod
  def get_name(cls):
    return "T-Test"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.Continuous]

  def _check_is_valid(self):
    A = self.groups[0]
    B = self.groups[1]
    warnings = [
      *self.check_normality(A),
      *self.check_normality(B),
    ]
    return _StatisticTestValidityModel(warnings=warnings)

  def significance(self):
    A = self.groups[0]
    B = self.groups[1]
    statistic, p_value = scipy.stats.ttest_ind(A, B)
    return SignificanceResult(
      type=StatisticTestMethodEnum.T,
      statistic=statistic, # type: ignore
      p_value=p_value # type: ignore
    )
  
class MannWhitneyUStatisticTest(_BaseStatisticTest):
  @classmethod
  def get_name(cls):
    return "Mann-Whitney U Test"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.OrderedCategorical, SchemaColumnTypeEnum.Temporal, SchemaColumnTypeEnum.Continuous]
  
  def check_normality(self, data: pd.Series)->list[str]:
    warnings = []
    if not pd.api.types.is_numeric_dtype(data.dtype):
      return warnings
    
    normaltest_result = scipy.stats.normaltest(data).pvalue
    is_normal = normaltest_result < 0.05
    if is_normal:
      warnings.append(f"Mann-Whitney U Test is generally used when the samples do not follow a normal distribution, but \"{data.name}\" does follow a normal distribution (p-value: {is_normal}). Consider using T-Test instead.")

    return warnings
  
  def _check_is_valid(self):
    A = self.groups[0]
    B = self.groups[1]
    warnings = [
      *self.check_normality(A),
      *self.check_normality(B),
    ]

    return _StatisticTestValidityModel(warnings=warnings)

  def significance(self):
    A, B = _mann_whitney_u_prepare(self.groups[0], self.groups[1])
    statistic, p_value = scipy.stats.mannwhitneyu(A, B)

    return SignificanceResult(
      type=StatisticTestMethodEnum.MannWhitneyU,
      statistic=statistic,
      p_value=p_value
    ) # type: ignore

class ChiSquaredStatisticTest(_BaseStatisticTest):
  type: Literal[StatisticTestMethodEnum.ChiSquared]

  @classmethod
  def get_name(cls):
    return "Chi-Squared Test"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.OrderedCategorical, SchemaColumnTypeEnum.Categorical, SchemaColumnTypeEnum.Topic]

  def _check_is_valid(self):
    contingency_table = _chisq_prepare(self.groups[0], self.groups[1], with_correction=False)
    less_than_5 = contingency_table < 5
    less_than_5_indices = np.argwhere(less_than_5)
    less_than_5_count: int = less_than_5.sum().sum()

    warnings: list[str] = []
    if less_than_5_count >= less_than_5.size * 0.2:
      observations: list[str] = []
      for index in less_than_5_indices:
        group_name = contingency_table.columns[index[1]]
        row_name = contingency_table.index[index[0]]
        observations.append(f"{group_name} - {row_name} ({contingency_table.iat[index[0], index[1]]} obs.)")

      warnings.append(f"More than 20% of the contingency table has less than 5 observations: {', '.join(observations[:5])}")

    return _StatisticTestValidityModel(warnings=warnings)
  
  def significance(self):
    res = scipy.stats.chi2_contingency(
      _chisq_prepare(self.groups[0], self.groups[1], with_correction=True)
    )
    return SignificanceResult(
      type=StatisticTestMethodEnum.ChiSquared,
      statistic=res.statistic, # type: ignore
      p_value=res.pvalue # type: ignore
    )
  
@dataclass
class StatisticTestFactory:
  column: SchemaColumn
  groups: list[pd.Series]
  preference: StatisticTestMethodEnum
  def build(self)->_BaseStatisticTest:
    if self.preference == StatisticTestMethodEnum.T:
      return TStatisticTest(column=self.column, groups=self.groups)
    elif self.preference == StatisticTestMethodEnum.MannWhitneyU:
      return MannWhitneyUStatisticTest(column=self.column, groups=self.groups)
    elif self.preference == StatisticTestMethodEnum.ChiSquared:
      return ChiSquaredStatisticTest(column=self.column, groups=self.groups)
    else:
      raise ValueError(f"\"{self.preference}\" is not a valid statistic test method.")


__all__ = [
  "StatisticTestFactory",
  "StatisticTestMethodEnum"
]