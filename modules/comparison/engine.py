from dataclasses import dataclass
import functools
import numpy as np
import pandas as pd
import pydantic

from modules.api.wrapper import ApiError
from modules.config import Config, SchemaColumn
from modules.project.cache import ProjectCacheManager
from modules.table import TableEngine, NamedTableFilter
from modules.logger import ProvisionedLogger

from .base import _StatisticTestValidityModel, SignificanceResult, EffectSizeResult
from .effect_size import EffectSizeFactory, EffectSizeMethodEnum
from .statistic_test import StatisticTestFactory, StatisticTestMethodEnum


logger = ProvisionedLogger().provision("TableComparisonEngine")

class TableComparisonGroupInfo(pydantic.BaseModel):
  name: str
  empty_count: int
  overlap_count: int
  valid_count: int
  total_count: int


class TableComparisonResult(pydantic.BaseModel):
  warnings: list[str]
  groups: list[TableComparisonGroupInfo]
  significance: SignificanceResult
  effect_size: EffectSizeResult


class TableComparisonEmptyException(Exception):
  pass

@dataclass
class TableComparisonEngine:
  config: Config
  engine: TableEngine
  groups: list[NamedTableFilter]
  exclude_overlapping_rows: bool = True
  
  @functools.cached_property
  def cache(self):
    return ProjectCacheManager().get(self.config.project_id)

  def _are_samples_large_enough(self, groups: list[pd.Series])->_StatisticTestValidityModel:
    warnings = []
    for gidx, group in enumerate(groups):
      if len(group) <= 10:
        warnings.append(f"\"{group.name}\" only has {len(group)} observations; this might not be enough to make definitive conclusions.")

    return _StatisticTestValidityModel(
      warnings=warnings
    )
  
  def _are_compared_groups_overlapping(self, groups: list[pd.Series])->_StatisticTestValidityModel:
    if len(groups) == 0:
      return _StatisticTestValidityModel()
    
    warnings: list[str] = []
    for i in range(len(groups)):
      for j in range(i + 1, len(groups)):
        data_A = groups[i].index
        data_B = groups[j].index
        overlap = data_A.intersection(data_B) # type: ignore
        if not overlap.empty:
          warnings.append(f"There are overlapping rows in \"{groups[i].name}\" and \"{groups[j].name}\". This may cause the statistic test to be unreliable. Considering adjusting the filter so that both groups are mutually exclusive.")
          
    return _StatisticTestValidityModel(
      warnings=warnings
    )

  def _exclude_overlapping_rows(self, groups: list[pd.Series], group_info: list[TableComparisonGroupInfo]):
    valid_mask = list(map(lambda group: np.full(len(group), 1, dtype=np.bool_), groups))
    for i in range(len(groups)):
      for j in range(i + 1, len(groups)):
        data_A = groups[i].index
        data_B = groups[j].index
        overlap = data_A.intersection(data_B) # type: ignore
        valid_mask[i][overlap] = 0
        valid_mask[j][overlap] = 0
    
    for i, mask in enumerate(valid_mask):
      groups[i] = groups[i][mask]
      overlap_count = int(len(mask) - mask.sum())
      group_info[i].overlap_count = overlap_count
      logger.info(f"Dropped {overlap_count} overlapping rows from {groups[i].name}.")

  def _exclude_na_rows(self, groups: list[pd.Series], group_info: list[TableComparisonGroupInfo]):
    for i in range(len(groups)):
      data = groups[i]
      notna_mask = data.notna()
      groups[i] = data[notna_mask]
      group_info[i].empty_count = int(len(data) - notna_mask.count())
      
  def check_is_valid(self, groups: list[pd.Series])->_StatisticTestValidityModel:
    validity1 = self._are_compared_groups_overlapping(groups)
    validity2 = self._are_samples_large_enough(groups)
    return validity1.merge(validity2)

  def preprocess(self, groups: list[pd.Series], column: SchemaColumn)->list[TableComparisonGroupInfo]:
    group_info = list(map(lambda group: TableComparisonGroupInfo(
      name=str(group.name),
      empty_count=0,
      overlap_count=0,
      valid_count=len(group),
      total_count=len(group),
    ), groups))
    if self.exclude_overlapping_rows:
      self._exclude_overlapping_rows(groups, group_info)
    self._exclude_na_rows(groups, group_info)
    for group in groups:
      if len(group) == 0:
        raise TableComparisonEmptyException(f"{group.name} does not have any values that can be compared.")

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
  
  def compare(self, df: pd.DataFrame, *, column_name: str, statistic_test_preference: StatisticTestMethodEnum, effect_size_preference: EffectSizeMethodEnum):
    column = self.config.data_schema.assert_exists(column_name)
    groups = self.load(df, column)
    group_info = self.preprocess(groups, column)
    validity = self.check_is_valid(groups)

    statistic_test_method = StatisticTestFactory(
      column=column,
      groups=groups,
      preference=statistic_test_preference
    ).build()

    effect_size_method = EffectSizeFactory(
      column=column,
      groups=groups,
      preference=effect_size_preference
    ).build()

    validity = statistic_test_method.check_is_valid()
    significance = statistic_test_method.significance()
    
    validity2 = effect_size_method.check_is_valid()
    validity.merge(validity2)
    effect_size = effect_size_method.effect_size()

    return TableComparisonResult(
      effect_size=effect_size,
      significance=significance,
      groups=group_info,
      warnings=validity.warnings
    )

__all__ = [
  "TableComparisonEngine",
  "TableComparisonResult",
  "TableComparisonGroupInfo",
  "TableComparisonEmptyException"
]