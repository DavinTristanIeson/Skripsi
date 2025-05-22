from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import scipy.stats

from modules.api import ExposedEnum
from modules.comparison.statistic_test import OmnibusStatisticTestFactory, OmnibusStatisticTestMethodEnum
from modules.comparison.utils import _chisq_prepare, _chisq_prepare_contingency_table, _mann_whitney_u_prepare, cramer_v
from modules.config import SchemaColumn, SchemaColumnTypeEnum
from modules.exceptions.dependencies import InvalidValueTypeException

from .base import _BaseEffectSize, EffectSizeResult, _StatisticTestValidityModel

class EffectSizeMethodEnum(str, Enum):
  MeanDifference = "mean-difference"
  MedianDifference = "median-difference"
  CohensD = "cohen-d"
  RankBiserialCorrelation = "rank-biserial-correlation"
  CramerV = "cramer-v"

class OmnibusEffectSizeMethodEnum(str, Enum):
  # For ANOVA
  EtaSquared = "eta-squared"
  # For Kruskal-Wallis
  EpsilonSquared = "epsilon-squared"
  CramerV = "cramer-v"

ExposedEnum().register(EffectSizeMethodEnum)
ExposedEnum().register(OmnibusEffectSizeMethodEnum)

class MeanDifferenceEffectSize(_BaseEffectSize):
  @classmethod
  def get_name(cls):
    return "Difference of Means"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.Continuous]
  
  def _check_is_valid(self):
    return _StatisticTestValidityModel()

  def effect_size(self):
    X = self.groups[0]
    Y = self.groups[1]
    effect = float(np.mean(X) - np.mean(Y))
    return EffectSizeResult(
      type=EffectSizeMethodEnum.MeanDifference,
      value=effect,
    )

class MedianDifferenceEffectSize(_BaseEffectSize):
  @classmethod
  def get_name(cls):
    return "Difference of Medians"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.Continuous]
  
  def _check_is_valid(self):
    return _StatisticTestValidityModel()
  
  def effect_size(self):
    X = self.groups[0]
    Y = self.groups[1]
    effect = float(np.median(X) - np.median(Y))
    return EffectSizeResult(
      type=EffectSizeMethodEnum.MedianDifference,
      value=effect,
    )

class CohenDEffectSize(_BaseEffectSize):
  @classmethod
  def get_name(cls):
    return "Cohen's D"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.Continuous]
  
  def _check_is_valid(self):
    return _StatisticTestValidityModel()
  
  def effect_size(self):
    X = self.groups[0]
    Y = self.groups[1]
    NX = len(X)
    NY = len(Y)

    group_mean_difference = np.mean(X) - np.mean(Y)
    pooled_stdev = np.sqrt(
      ((NX - 1) * np.var(X, ddof=1)) + ((NY - 1) * np.var(Y, ddof=1)) /
      (NX + NY)
    )
    cohen_d = group_mean_difference / pooled_stdev
    return EffectSizeResult(
      type=EffectSizeMethodEnum.CohensD,
      value=cohen_d,
    )

class RankBiserialEffectSize(_BaseEffectSize):
  @classmethod
  def get_name(cls):
    return "Rank Biserial Correlation"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.OrderedCategorical, SchemaColumnTypeEnum.Temporal, SchemaColumnTypeEnum.Continuous]
  
  def _check_is_valid(self):
    return _StatisticTestValidityModel()

  def effect_size(self):
    groups = _mann_whitney_u_prepare(self.groups)
    A = groups[0]
    B = groups[1]
    full_data = np.hstack([A, B])
    ranks = scipy.stats.rankdata(full_data)
    ranks_a = ranks[:len(A)]
    ranks_b = ranks[len(A):]
    rank_biserial = 2 * (ranks_a.mean() - ranks_b.mean()) / len(full_data)
    return EffectSizeResult(
      type=EffectSizeMethodEnum.RankBiserialCorrelation,
      value=rank_biserial,
    )
  
class CramerVEffectSize(_BaseEffectSize):
  @classmethod
  def get_name(cls):
    return "Cramer's V"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.OrderedCategorical, SchemaColumnTypeEnum.Categorical, SchemaColumnTypeEnum.Topic]
  
  def _check_is_valid(self):
    return _StatisticTestValidityModel()

  def effect_size(self):
    contingency_table = _chisq_prepare(self.groups, with_correction=True)
    V = cramer_v(contingency_table)
    
    # V = scipy.stats.contingency.association(contingency_table, method="cramer")
    return EffectSizeResult(
      type=EffectSizeMethodEnum.CramerV,
      value=V,
    )
  
  def effect_size_contingency(self, contingency_table: pd.DataFrame):
    contingency_table = _chisq_prepare_contingency_table(contingency_table, with_correction=True)
    V = cramer_v(contingency_table)
    
    # V = scipy.stats.contingency.association(contingency_table, method="cramer")
    return EffectSizeResult(
      type=EffectSizeMethodEnum.CramerV,
      value=V,
    )


@dataclass
class EffectSizeFactory:
  column: SchemaColumn
  groups: list[pd.Series]
  preference: EffectSizeMethodEnum
  def build(self)->_BaseEffectSize:
    if self.preference == EffectSizeMethodEnum.MeanDifference:
      return MeanDifferenceEffectSize(column=self.column, groups=self.groups)
    elif self.preference == EffectSizeMethodEnum.MedianDifference:
      return MedianDifferenceEffectSize(column=self.column, groups=self.groups)
    elif self.preference == EffectSizeMethodEnum.CohensD:
      return CohenDEffectSize(column=self.column, groups=self.groups)
    elif self.preference == EffectSizeMethodEnum.RankBiserialCorrelation:
      return RankBiserialEffectSize(column=self.column, groups=self.groups)
    elif self.preference == EffectSizeMethodEnum.CramerV:
      return CramerVEffectSize(column=self.column, groups=self.groups)
    else:
      raise InvalidValueTypeException(type="effect size", value=self.preference)
    
class EtaSquaredEffectSize(_BaseEffectSize):
  @classmethod
  def get_name(cls):
    return "Eta-Squared"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.Continuous]
  
  def _check_is_valid(self):
    return _StatisticTestValidityModel()

  def effect_size(self):
    # https://stackoverflow.com/questions/52083501/how-to-compute-correlation-ratio-or-eta-in-python
    ssw = 0
    ssb = 0
    group_total = sum(map(np.sum, self.groups))
    group_length = sum(map(len, self.groups))
    global_mean = group_total / group_length
    for group in self.groups:
      local_mean = group.mean()
      ssw += np.power(group - local_mean, 2).sum()
      ssb += len(group) * np.power(local_mean - global_mean, 2)
    eta_squared = ssb / (ssb + ssw)

    return EffectSizeResult(
      type=OmnibusEffectSizeMethodEnum.EtaSquared,
      value=eta_squared,
    )
    
class EpsilonSquaredEffectSize(_BaseEffectSize):
  @classmethod
  def get_name(cls):
    return "Epsilon-Squared"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.Temporal, SchemaColumnTypeEnum.Continuous, SchemaColumnTypeEnum.OrderedCategorical]
  
  def _check_is_valid(self):
    return _StatisticTestValidityModel()

  def effect_size(self):
    groups = _mann_whitney_u_prepare(self.groups)
    H, p_value = scipy.stats.kruskal(*groups)

    # https://www.researchgate.net/post/Anyone-know-how-to-calculate-eta-squared-for-a-Kruskal-Wallis-analysis
    N = sum(map(len, groups))
    epsilon_squared = H / (
      (pow(N, 2) - 1) /
      (N + 1)
    )
    return EffectSizeResult(
      type=OmnibusEffectSizeMethodEnum.EpsilonSquared,
      value=epsilon_squared,
    )
    
@dataclass
class OmnibusEffectSizeFactory:
  column: SchemaColumn
  groups: list[pd.Series]
  preference: OmnibusEffectSizeMethodEnum
  def build(self)->_BaseEffectSize:
    if self.preference == OmnibusEffectSizeMethodEnum.EtaSquared:
      return EtaSquaredEffectSize(column=self.column, groups=self.groups)
    elif self.preference == OmnibusEffectSizeMethodEnum.EpsilonSquared:
      return EpsilonSquaredEffectSize(column=self.column, groups=self.groups)
    elif self.preference == OmnibusEffectSizeMethodEnum.CramerV:
      return CramerVEffectSize(column=self.column, groups=self.groups)
    else:
      raise InvalidValueTypeException(type="effect size", value=self.preference)
    
  @staticmethod
  def from_statistic_test(factory: OmnibusStatisticTestFactory):
    if factory.preference == OmnibusStatisticTestMethodEnum.ANOVA:
      return EtaSquaredEffectSize(column=factory.column, groups=factory.groups)
    elif factory.preference == OmnibusStatisticTestMethodEnum.KruskalWallis:
      return EpsilonSquaredEffectSize(column=factory.column, groups=factory.groups)
    elif factory.preference == OmnibusStatisticTestMethodEnum.ChiSquared:
      return CramerVEffectSize(column=factory.column, groups=factory.groups)
    else:
      raise InvalidValueTypeException(type="statistic method", value=factory.preference)
    
    
__all__ = [
  "EffectSizeFactory",
  "EffectSizeMethodEnum",
  "OmnibusEffectSizeFactory",
  "OmnibusEffectSizeMethodEnum",
]