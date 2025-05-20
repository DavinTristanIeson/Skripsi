from dataclasses import dataclass
import functools
import pandas as pd
import pydantic

from modules.comparison.exceptions import EmptyComparisonGroupException, NotMutuallyExclusiveException
from modules.comparison.utils import assert_mutually_exclusive
from modules.config import Config, SchemaColumn
from modules.project.cache_manager import ProjectCacheManager
from modules.table import TableEngine, NamedTableFilter
from modules.logger import ProvisionedLogger

from .base import _StatisticTestValidityModel, SignificanceResult, EffectSizeResult
from .effect_size import EffectSizeFactory, EffectSizeMethodEnum, GroupEffectSizeFactory
from .statistic_test import GroupStatisticTestFactory, GroupStatisticTestMethodEnum, StatisticTestFactory, StatisticTestMethodEnum


logger = ProvisionedLogger().provision("TableComparisonEngine")

class TableComparisonGroupInfo(pydantic.BaseModel):
  name: str
  empty_count: int
  valid_count: int
  total_count: int

class StatisticTestResult(pydantic.BaseModel):
  warnings: list[str]
  groups: list[TableComparisonGroupInfo]
  significance: SignificanceResult
  effect_size: EffectSizeResult

@dataclass
class TableComparisonPreprocessResult:
  column: SchemaColumn
  groups: list[pd.Series]
  group_info: list[TableComparisonGroupInfo]
  

@dataclass
class TableComparisonEngine:
  config: Config
  groups: list[NamedTableFilter]
  
  @functools.cached_property
  def cache(self):
    return ProjectCacheManager().get(self.config.project_id)

  @functools.cached_property
  def engine(self):
    return TableEngine(
      config=self.config,
    )

  def _exclude_na_rows(self, groups: list[pd.Series], group_info: list[TableComparisonGroupInfo]):
    for i in range(len(groups)):
      data = groups[i]
      notna_mask = data.notna()
      groups[i] = data[notna_mask]
      group_info[i].empty_count = int(len(data) - notna_mask.count())

  def preprocess(self, groups: list[pd.Series], column: SchemaColumn)->list[TableComparisonGroupInfo]:
    group_info = list(map(lambda group: TableComparisonGroupInfo(
      name=str(group.name),
      empty_count=0,
      valid_count=len(group),
      total_count=len(group),
    ), groups))
    assert_mutually_exclusive(groups)
    self._exclude_na_rows(groups, group_info)
    for group in groups:
      if len(group) == 0:
        raise EmptyComparisonGroupException(group=str(group.name))

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
    group_info = self.preprocess(groups, column)
    return TableComparisonPreprocessResult(
      group_info=group_info,
      groups=groups,
      column=column,
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
    effect_size = effect_size_method.effect_size()

    return StatisticTestResult(
      effect_size=effect_size,
      significance=significance,
      groups=preprocess_result.group_info,
      warnings=validity.warnings
    )
  
  def compare_group(self, df: pd.DataFrame, *, column_name: str, statistic_test_preference: GroupStatisticTestMethodEnum):
    preprocess_result = self.extract_groups(df, column_name)

    statistic_test_method_factory = GroupStatisticTestFactory(
      column=preprocess_result.column,
      groups=preprocess_result.groups,
      preference=statistic_test_preference
    )
    statistic_test_method = statistic_test_method_factory.build()
    effect_size_method = GroupEffectSizeFactory.from_statistic_test(statistic_test_method_factory)

    validity = statistic_test_method.check_is_valid()
    significance = statistic_test_method.significance()
    
    validity2 = effect_size_method.check_is_valid()
    validity.merge(validity2)
    effect_size = effect_size_method.effect_size()

    return StatisticTestResult(
      effect_size=effect_size,
      significance=significance,
      groups=preprocess_result.group_info,
      warnings=validity.warnings
    )

__all__ = [
  "TableComparisonEngine",
  "StatisticTestResult",
  "TableComparisonGroupInfo",
]