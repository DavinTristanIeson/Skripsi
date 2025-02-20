from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import scipy.stats

from modules.api import ExposedEnum
from modules.config import SchemaColumn, SchemaColumnTypeEnum

from .base import BaseEffectSize, EffectSizeResult, SignificanceResult, StatisticTestValidityModel
from .statistic_test import StatisticTestMethodEnum

class EffectSizeMethodEnum(str, Enum):
  Auto = "auto"
  MeanDifference = "mean-difference"
  MedianDifference = "median-difference"
  CohensD = "cohen-d"
  RankBiserialCorrelation = "rank-biserial-correlation"
  CramerV = "cramer-v"

ExposedEnum().register(EffectSizeMethodEnum)

class MeanDifferenceEffectSize(BaseEffectSize):
  @classmethod
  def get_name(cls):
    return "Difference of Means"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.Continuous]
  
  def _check_is_valid(self):
    return StatisticTestValidityModel()

  def effect_size(self):
    X = self.groups[0]
    Y = self.groups[1]
    effect = float(np.mean(X) - np.mean(Y))
    return EffectSizeResult(
      type=EffectSizeMethodEnum.MeanDifference,
      value=effect,
    )

class MedianDifferenceEffectSize(BaseEffectSize):
  @classmethod
  def get_name(cls):
    return "Difference of Medians"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.Continuous]
  
  def _check_is_valid(self):
    return StatisticTestValidityModel()
  
  def effect_size(self):
    X = self.groups[0]
    Y = self.groups[1]
    effect = float(np.median(X) - np.median(Y))
    return EffectSizeResult(
      type=EffectSizeMethodEnum.MedianDifference,
      value=effect,
    )

class CohenDEffectSize(BaseEffectSize):
  @classmethod
  def get_name(cls):
    return "Cohen's D"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.Continuous]
  
  def _check_is_valid(self):
    warnings = [
      *self.check_normality(self.groups[0]),
      *self.check_normality(self.groups[1]),
    ]
    return StatisticTestValidityModel(warnings=warnings)
  
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

class RankBiserialEffectSize(BaseEffectSize):
  @classmethod
  def get_name(cls):
    return "Rank Biserial Correlation"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.OrderedCategorical, SchemaColumnTypeEnum.Temporal, SchemaColumnTypeEnum.Continuous]
  
  def _check_is_valid(self):
    return StatisticTestValidityModel()

  def effect_size(self):
    X = self.groups[0]
    Y = self.groups[1]
    NX = len(X)
    NY = len(Y)

    U, pvalue = scipy.stats.mannwhitneyu(X, Y)
    # A bit silly that we have to run a Mann Whitney U Test again, but oh well.
    # It's better than making a mistake with the actual full formula.
    # Based on https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Rank-biserial_correlation
    rb = (2 * U) / (NX * NY) - 1
    return EffectSizeResult(
      type=EffectSizeMethodEnum.RankBiserialCorrelation,
      value=rb,
    )
  
class CramerVEffectSize(BaseEffectSize):
  @classmethod
  def get_name(cls):
    return "Cramer's V"
  
  @classmethod
  def get_supported_types(cls):
    return [SchemaColumnTypeEnum.OrderedCategorical, SchemaColumnTypeEnum.Categorical, SchemaColumnTypeEnum.MultiCategorical, SchemaColumnTypeEnum.Topic]
  
  def _check_is_valid(self):
    return StatisticTestValidityModel()
  
  
  def contingency_table(self):
    A = self.groups[0]
    B = self.groups[1]
    A_freq = A.value_counts()
    B_freq = B.value_counts()
    crosstab = pd.concat([A_freq, B_freq], axis=0)
    crosstab.fillna(0, inplace=True)
    return crosstab

  def effect_size(self):
    contingency_table = self.contingency_table()
    V = scipy.stats.contingency.association(contingency_table, method="cramer")
    return EffectSizeResult(
      type=EffectSizeMethodEnum.CramerV,
      value=V,
    )

@dataclass
class EffectSizeFactory:
  column: SchemaColumn
  groups: list[pd.Series]
  preference: EffectSizeMethodEnum
  def build(self, result: SignificanceResult)->BaseEffectSize:
    mean_difference = MeanDifferenceEffectSize(column=self.column, groups=self.groups)
    median_difference = MedianDifferenceEffectSize(column=self.column, groups=self.groups)
    cohens_d = CohenDEffectSize(column=self.column, groups=self.groups)
    rank_biserial = RankBiserialEffectSize(column=self.column, groups=self.groups)
    cramer_v = CramerVEffectSize(column=self.column, groups=self.groups)

    if self.preference == EffectSizeMethodEnum.MeanDifference:
      return mean_difference
    elif self.preference == EffectSizeMethodEnum.MedianDifference:
      return median_difference
    elif self.preference == EffectSizeMethodEnum.CohensD:
      return cohens_d
    elif self.preference == EffectSizeMethodEnum.RankBiserialCorrelation:
      return rank_biserial
    elif self.preference == EffectSizeMethodEnum.CramerV:
      return cramer_v
    
    if result.type == StatisticTestMethodEnum.ChiSquared:
      return cramer_v
    elif result.type == StatisticTestMethodEnum.MannWhitneyU:
      return rank_biserial
    elif result.type == StatisticTestMethodEnum.T:
      return cohens_d
    else:
      raise ValueError(f"Column of type \"{self.column.type}\" cannot be compared.")
    
__all__ = [
  "EffectSizeFactory"
]