from dataclasses import dataclass
from enum import Enum
import itertools
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats

from modules.comparison.utils import _check_chisq_contingency_table, _check_non_normal_distribution, _check_normal_distribution, _chisq_prepare, _chisq_prepare_contingency_table, _mann_whitney_u_prepare
from modules.logger import ProvisionedLogger
from modules.api import ExposedEnum
from modules.config import SchemaColumn, SchemaColumnTypeEnum
from .base import _BaseStatisticTest, SignificanceResult, _StatisticTestValidityModel

logger = ProvisionedLogger().provision("TableComparisonEngine")

class StatisticTestMethodEnum(str, Enum):
  T = "t"
  MannWhitneyU = "mann-whitney-u"
  ChiSquared = "chi-squared"

class GroupStatisticTestMethodEnum(str, Enum):
  ANOVA = "anova"
  KruskalWallis = "kruskal-wallis"
  ChiSquared = "chi-squared"

ExposedEnum().register(StatisticTestMethodEnum)
ExposedEnum().register(GroupStatisticTestMethodEnum)

class TStatisticTest(_BaseStatisticTest):
  @classmethod
  def get_name(cls):
    return "T-Test"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.Continuous]

  def _check_is_valid(self):
    return _StatisticTestValidityModel(
      warnings=_check_normal_distribution(name=self.get_name(), groups=self.groups)
    )

  def significance(self):
    A = self.groups[0]
    B = self.groups[1]
    statistic, p_value = scipy.stats.ttest_ind(A, B, nan_policy="omit")
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
  
  def _check_is_valid(self):
    return _StatisticTestValidityModel(
      warnings=_check_non_normal_distribution(name=self.get_name(), groups=self.groups)
    )

  def significance(self):
    groups = _mann_whitney_u_prepare(self.groups)
    statistic, p_value = scipy.stats.mannwhitneyu(groups[0], groups[1])

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
    contingency_table = _chisq_prepare(self.groups, with_correction=False)
    return _StatisticTestValidityModel(warnings=_check_chisq_contingency_table(contingency_table))
  
  def significance(self):
    res = scipy.stats.chi2_contingency(
      _chisq_prepare(self.groups, with_correction=True),
    )
    return SignificanceResult(
      type=StatisticTestMethodEnum.ChiSquared,
      statistic=res.statistic, # type: ignore
      p_value=res.pvalue # type: ignore
    )

  def significance_contingency(self, contingency_table: pd.DataFrame):
    res = scipy.stats.chi2_contingency(
      _chisq_prepare_contingency_table(contingency_table, with_correction=True),
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
    
class ANOVAStatisticTest(_BaseStatisticTest):
  @classmethod
  def get_name(cls):
    return "One-Way ANOVA F-Test"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.Continuous]

  def _check_is_valid(self):
    return _StatisticTestValidityModel(
      warnings=_check_normal_distribution(groups=self.groups, name=self.get_name())
    )

  def significance(self):
    statistic, p_value = scipy.stats.f_oneway(*self.groups)
    return SignificanceResult(
      type=GroupStatisticTestMethodEnum.ANOVA,
      statistic=statistic, # type: ignore
      p_value=p_value # type: ignore
    )
  

class KruskalWallisStatisticTest(_BaseStatisticTest):
  @classmethod
  def get_name(cls):
    return "Kruskal-Wallis H Test"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.Temporal, SchemaColumnTypeEnum.Continuous, SchemaColumnTypeEnum.OrderedCategorical]

  def _check_is_valid(self):
    return _StatisticTestValidityModel(
      warnings=_check_non_normal_distribution(groups=self.groups, name=self.get_name())
    )

  def significance(self):
    groups = _mann_whitney_u_prepare(self.groups)
    statistic, p_value = scipy.stats.kruskal(*groups)
    return SignificanceResult(
      type=GroupStatisticTestMethodEnum.KruskalWallis,
      statistic=statistic, # type: ignore
      p_value=p_value # type: ignore
    )

@dataclass
class GroupStatisticTestFactory:
  column: SchemaColumn
  groups: list[pd.Series]
  preference: GroupStatisticTestMethodEnum
  def build(self)->_BaseStatisticTest:
    if self.preference == GroupStatisticTestMethodEnum.ANOVA:
      return ANOVAStatisticTest(column=self.column, groups=self.groups)
    elif self.preference == GroupStatisticTestMethodEnum.KruskalWallis:
      return KruskalWallisStatisticTest(column=self.column, groups=self.groups)
    elif self.preference == GroupStatisticTestMethodEnum.ChiSquared:
      return ChiSquaredStatisticTest(column=self.column, groups=self.groups)
    else:
      raise ValueError(f"\"{self.preference}\" is not a valid statistic test method.")
   

__all__ = [
  "StatisticTestFactory",
  "StatisticTestMethodEnum"
]