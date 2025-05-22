from dataclasses import dataclass
import functools
from typing import Optional
import pandas as pd
import pydantic

from modules.comparison.exceptions import EmptyComparisonGroupException
from modules.comparison.utils import assert_mutually_exclusive
from modules.config import Config, SchemaColumn
from modules.project.cache_manager import ProjectCacheManager
from modules.table import TableEngine, NamedTableFilter
from modules.logger import ProvisionedLogger

from .base import _StatisticTestValidityModel, SignificanceResult, EffectSizeResult
from .effect_size import EffectSizeFactory, EffectSizeMethodEnum, OmnibusEffectSizeFactory
from .statistic_test import OmnibusStatisticTestFactory, OmnibusStatisticTestMethodEnum, StatisticTestFactory, StatisticTestMethodEnum


logger = ProvisionedLogger().provision("TableComparisonEngine")

class ComparisonGroupInfo(pydantic.BaseModel):
  name: str
  empty_count: int
  valid_count: int
  total_count: int
  overlap_count: int

class StatisticTestResult(pydantic.BaseModel):
  warnings: list[str]
  groups: list[ComparisonGroupInfo]
  significance: SignificanceResult
  effect_size: EffectSizeResult
  sample_size: int

@dataclass
class TableComparisonPreprocessResult:
  column: SchemaColumn
  groups: list[pd.Series]
  group_info: list[ComparisonGroupInfo]
  validity: _StatisticTestValidityModel

@dataclass
class TableComparisonEngine:
  config: Config
  groups: list[NamedTableFilter]
  exclude_overlapping_rows: bool
  
  @functools.cached_property
  def cache(self):
    return ProjectCacheManager().get(self.config.project_id)

  @functools.cached_property
  def engine(self):
    return TableEngine(
      config=self.config,
    )
  
  def _are_samples_large_enough(self, groups: list[pd.Series], validity: _StatisticTestValidityModel):
    for gidx, group in enumerate(groups):
      if len(group) <= 10:
        validity.warnings.append(f"\"{group.name}\" only has {len(group)} observations; this might not be enough to make definitive conclusions.")

  def _are_compared_groups_overlapping(self, groups: list[pd.Series], group_info: list[ComparisonGroupInfo], validity: _StatisticTestValidityModel):
    # Use a pandas mask so that the indices are preserved
    valid_mask = list(map(lambda group: pd.Series(data=True, index=group.index), groups))
    for i in range(len(groups)):
      for j in range(i + 1, len(groups)):
        data_A = groups[i].index
        data_B = groups[j].index
        overlap = data_A.intersection(data_B) # type: ignore
        valid_mask[i][overlap] = False
        valid_mask[j][overlap] = False
        if not overlap.empty:
          validity.warnings.append(f"There are {len(overlap)} overlapping rows in \"{groups[i].name}\" and \"{groups[j].name}\". This may cause the statistic test to be unreliable. Considering adjusting the filter so that both groups are mutually exclusive.")
    
    for i, mask in enumerate(valid_mask):
      overlap_count = int(len(mask) - mask.sum())
      group_info[i].overlap_count = overlap_count
      group_info[i].valid_count -= overlap_count
      if self.exclude_overlapping_rows:
        groups[i] = groups[i][mask]
        logger.info(f"Dropped {overlap_count} overlapping rows from {groups[i].name}.")


  def _exclude_na_rows(self, groups: list[pd.Series], group_info: list[ComparisonGroupInfo]):
    for i in range(len(groups)):
      data = groups[i]
      notna_mask = data.notna()
      not_empty_count = notna_mask.sum()
      groups[i] = data[notna_mask]
      group_info[i].empty_count = int(len(data) - not_empty_count)
      group_info[i].valid_count = not_empty_count

  def preprocess(self, groups: list[pd.Series], *, total_count: Optional[int] = None, validity: _StatisticTestValidityModel)->list[ComparisonGroupInfo]:
    if total_count is None:
      total_count = max(map(len, groups))
    group_info = list(map(lambda group: ComparisonGroupInfo(
      name=str(group.name),
      empty_count=0,
      valid_count=len(group),
      total_count=total_count,
      overlap_count=0,
    ), groups))
    self._exclude_na_rows(groups, group_info)
    self._are_compared_groups_overlapping(groups, group_info, validity)
    self._are_samples_large_enough(groups, validity)
    for group in groups:
      if len(group) == 0:
        raise EmptyComparisonGroupException(
          group=str(group.name),
          exclude_overlapping_rows=self.exclude_overlapping_rows
        )

    for group, ginfo in zip(groups, group_info):
      ginfo.valid_count = len(group)

    return group_info

  def load(self, df: pd.DataFrame, column: SchemaColumn)->list[pd.Series]:
    data_groups: list[pd.Series] = []
    for group in self.groups:
      filtered_df = self.engine.filter(df, group.filter)
      data = filtered_df[column.name]
      data.name = group.name
      data_groups.append(data)
    return data_groups
  
  def extract_groups(self, df: pd.DataFrame, column_name: str):
    column = self.config.data_schema.assert_exists(column_name)
    groups = self.load(df, column)
    validity = _StatisticTestValidityModel()
    group_info = self.preprocess(groups, total_count=len(df), validity=validity)
    return TableComparisonPreprocessResult(
      group_info=group_info,
      groups=groups,
      column=column,
      validity=validity,
    )
  
  def compare(self, df: pd.DataFrame, *, column_name: str, statistic_test_preference: StatisticTestMethodEnum, effect_size_preference: EffectSizeMethodEnum):
    preprocess_result = self.extract_groups(df, column_name)

    statistic_test_method = StatisticTestFactory(
      column=preprocess_result.column,
      groups=preprocess_result.groups,
      preference=statistic_test_preference
    ).build()

    effect_size_method = EffectSizeFactory(
      column=preprocess_result.column,
      groups=preprocess_result.groups,
      preference=effect_size_preference
    ).build()

    validity = statistic_test_method.check_is_valid()
    significance = statistic_test_method.significance()
    
    validity2 = effect_size_method.check_is_valid()
    validity.merge(validity2)
    validity.merge(preprocess_result.validity)
    effect_size = effect_size_method.effect_size()

    return StatisticTestResult(
      effect_size=effect_size,
      significance=significance,
      groups=preprocess_result.group_info,
      warnings=validity.warnings,
      sample_size=sum(map(len, preprocess_result.groups)),
    )
  
  def compare_omnibus(self, df: pd.DataFrame, *, column_name: str, statistic_test_preference: OmnibusStatisticTestMethodEnum):
    preprocess_result = self.extract_groups(df, column_name)

    statistic_test_method_factory = OmnibusStatisticTestFactory(
      column=preprocess_result.column,
      groups=preprocess_result.groups,
      preference=statistic_test_preference
    )
    statistic_test_method = statistic_test_method_factory.build()
    effect_size_method = OmnibusEffectSizeFactory.from_statistic_test(statistic_test_method_factory)

    validity = statistic_test_method.check_is_valid()
    significance = statistic_test_method.significance()
    
    validity2 = effect_size_method.check_is_valid()
    validity.merge(validity2)
    validity.merge(preprocess_result.validity)
    effect_size = effect_size_method.effect_size()

    return StatisticTestResult(
      effect_size=effect_size,
      significance=significance,
      groups=preprocess_result.group_info,
      warnings=validity.warnings,
      sample_size=sum(map(len, preprocess_result.groups))
    )

__all__ = [
  "TableComparisonEngine",
  "StatisticTestResult",
  "ComparisonGroupInfo",
]