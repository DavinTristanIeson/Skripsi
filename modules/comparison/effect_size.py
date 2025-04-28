from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import scipy.stats

from modules.api import ExposedEnum
from modules.comparison.utils import _chisq_prepare, _mann_whitney_u_prepare
from modules.config import SchemaColumn, SchemaColumnTypeEnum

from .base import _BaseEffectSize, EffectSizeResult, _StatisticTestValidityModel

class EffectSizeMethodEnum(str, Enum):
  MeanDifference = "mean-difference"
  MedianDifference = "median-difference"
  CohensD = "cohen-d"
  RankBiserialCorrelation = "rank-biserial-correlation"
  CramerV = "cramer-v"

ExposedEnum().register(EffectSizeMethodEnum)

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

    # Cramer V with bias correction (https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V)
    chi2_result = scipy.stats.chi2_contingency(contingency_table)
    chi2 = chi2_result.statistic # type: ignore

    n = contingency_table.sum().sum()
    psi2 = chi2 / n
    k = contingency_table.shape[1]
    r = contingency_table.shape[0]

    k_tilde = k - (np.power(k - 1, 2) / (n - 1))
    r_tilde = r - (np.power(r - 1, 2) / (n - 1))
    psi2_tilde = max(0, psi2 - ((k-1) * (r-1) / (n-1)))

    V = np.sqrt(psi2_tilde / min(k_tilde - 1, r_tilde - 1))
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
      raise ValueError(f"\"{self.preference}\" is not a valid effect size.")
    
__all__ = [
  "EffectSizeFactory",
  "EffectSizeMethodEnum"
]