from dataclasses import dataclass
from enum import Enum
import http
import itertools
from typing import Literal, cast

import numpy as np
import pandas as pd

from common.logger import RegisteredLogger
from common.models.api import ApiError
from common.models.enum import ExposedEnum
from models.comparison.base import BaseStatisticTest, SignificanceResult, StatisticTestValidityModel

import scipy.stats

from models.comparison.effect_size import CohenDEffectSize
from models.config.schema import CategoricalSchemaColumn, SchemaColumn, SchemaColumnTypeEnum

logger = RegisteredLogger().provision("TableComparisonEngine")

class StatisticTestMethodEnum(str, Enum):
  Auto = "auto"
  T = "t"
  MannWhitneyU = "mann-whitney-u"
  ChiSquared = "chi-squared"

ExposedEnum().register(StatisticTestMethodEnum)

class TStatisticTest(BaseStatisticTest):
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
    return StatisticTestValidityModel(warnings=warnings)

  def significance(self):
    A = self.groups[0]
    B = self.groups[1]
    statistic, p_value = scipy.stats.ttest_ind(A, B)
    return SignificanceResult(
      type=StatisticTestMethodEnum.T,
      statistic=statistic, # type: ignore
      p_value=p_value # type: ignore
    )
  
class MannWhitneyUStatisticTest(BaseStatisticTest):
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

    return StatisticTestValidityModel(warnings=warnings)

  def significance(self):
    A = self.groups[0]
    B = self.groups[1]
    statistic, p_value = scipy.stats.mannwhitneyu(A, B)

    return SignificanceResult(
      type=StatisticTestMethodEnum.MannWhitneyU,
      statistic=statistic,
      p_value=p_value
    ) # type: ignore

class ChiSquaredStatisticTest(BaseStatisticTest):
  type: Literal[StatisticTestMethodEnum.ChiSquared]

  @classmethod
  def get_name(cls):
    return "Chi-Squared Test"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.OrderedCategorical, SchemaColumnTypeEnum.Categorical, SchemaColumnTypeEnum.MultiCategorical, SchemaColumnTypeEnum.Topic]

  def contingency_table(self):
    A = self.groups[0]
    B = self.groups[1]
    A_freq = A.value_counts()
    B_freq = B.value_counts()
    crosstab = pd.concat([A_freq, B_freq], axis=0)
    crosstab.fillna(0, inplace=True)
    return crosstab

  def _check_is_valid(self):
    contingency_table = self.contingency_table()
    less_than_5 = contingency_table < 5
    less_than_5_indices = np.argwhere(less_than_5)
    less_than_5_count: int = less_than_5.sum() # type: ignore

    warnings: list[str] = []
    if less_than_5_count >= less_than_5.size * 0.2:
      observations = itertools.islice(map(lambda index: f"({contingency_table.index[index[0]]} x {contingency_table.columns[index[1]]}: {contingency_table.iloc[index]} obs.)", less_than_5_indices), 5)
      warnings.append(f"More than 20% of the contingency table has less than 5 observations: {', '.join(observations)}")

    return StatisticTestValidityModel(warnings=warnings)
  
  def significance(self):
    statistic, p_value = scipy.stats.chi2_contingency(
      self.contingency_table()
    )
    return SignificanceResult(
      type=StatisticTestMethodEnum.ChiSquared,
      statistic=statistic, # type: ignore
      p_value=p_value # type: ignore
    )
  
@dataclass
class StatisticTestFactory:
  column: SchemaColumn
  groups: list[pd.Series]
  preference: StatisticTestMethodEnum
  def build(self)->BaseStatisticTest:
    t = TStatisticTest(column=self.column, groups=self.groups)
    mann_whitney_u = MannWhitneyUStatisticTest(column=self.column, groups=self.groups)
    chi_squared = ChiSquaredStatisticTest(column=self.column, groups=self.groups)

    if self.preference == StatisticTestMethodEnum.T:
      return t
    elif self.preference == StatisticTestMethodEnum.MannWhitneyU:
      return mann_whitney_u
    elif self.preference == StatisticTestMethodEnum.ChiSquared:
      return chi_squared
    
    # Auto
    if self.column.type == SchemaColumnTypeEnum.Categorical:
      return chi_squared
    elif self.column.type == SchemaColumnTypeEnum.OrderedCategorical:
      return mann_whitney_u
    elif self.column.type == SchemaColumnTypeEnum.Continuous:
      are_normals = list(map(
        lambda group: scipy.stats.normaltest(group).pvalue < 0.05,
        self.groups
      ))
      if all(are_normals):
        return t
      else:
        return mann_whitney_u
    elif self.column.type == SchemaColumnTypeEnum.Topic:
      return chi_squared
    elif self.column.type == SchemaColumnTypeEnum.Temporal:
      return mann_whitney_u
    elif self.column.type == SchemaColumnTypeEnum.MultiCategorical:
      return chi_squared
    else:
      raise ValueError(f"Column of type \"{self.column.type}\" cannot be compared.")

__all__ = [
  "StatisticTestFactory"
]