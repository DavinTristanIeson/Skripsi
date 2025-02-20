from dataclasses import dataclass
import pandas as pd

from modules.config import Config, SchemaColumn
from modules.table import TableEngine, NamedTableFilter

from modules.logger import ProvisionedLogger
from .api import TableComparisonGroupInfoResource, TableComparisonResource
from .base import StatisticTestValidityModel
from .effect_size import EffectSizeFactory, EffectSizeMethodEnum
from .statistic_test import StatisticTestFactory, StatisticTestMethodEnum


logger = ProvisionedLogger().provision("TableComparisonEngine")

@dataclass
class TableComparisonEngine:
  config: Config
  engine: TableEngine
  groups: list[NamedTableFilter]
  exclude_overlapping_rows: bool = True

  def __init__(self, config: Config) -> None:
    self.config = config
    self.engine = TableEngine(config)

  def _are_samples_large_enough(self, groups: list[pd.Series])->StatisticTestValidityModel:
    warnings = []
    for gidx, group in enumerate(groups):
      if len(group) <= 10:
        warnings.append(f"\"{group.name}\" only has {len(group)} observations; this might not be enough to make definitive conclusions.")

    return StatisticTestValidityModel(
      warnings=warnings
    )
  
  def _are_compared_groups_overlapping(self, groups: list[pd.Series])->StatisticTestValidityModel:
    if len(groups) == 0:
      return StatisticTestValidityModel()
    
    warnings: list[str] = []
    for i in range(len(groups)):
      for j in range(i + 1, len(groups)):
        data_A = groups[i].index
        data_B = groups[j].index
        overlap = data_A.intersection(data_B) # type: ignore
        if not overlap.empty:
          warnings.append(f"There are overlapping rows in \"{groups[i].name}\" and \"{groups[j].name}\". This may cause the statistic test to be unreliable. Considering adjusting the filter so that both groups are mutually exclusive.")
          
    return StatisticTestValidityModel(
      warnings=warnings
    )

  def _exclude_overlapping_rows(self, groups: list[pd.Series]):
    for i in range(len(groups)):
      for j in range(i + 1, len(groups)):
        data_A = groups[i].index
        data_B = groups[j].index
        overlap = data_A.intersection(data_B) # type: ignore
        groups[i].drop(overlap, axis=0, inplace=True)
        groups[j].drop(overlap, axis=0, inplace=True)
        logger.info(f"Dropped {len(overlap)} rows from {groups[i].name} and {groups[j].name} because they overlap.")

  def _exclude_na_rows(self, groups: list[pd.Series])->list[int]:
    invalid_count = []
    for i in range(len(groups)):
      data = groups[i]
      notna_mask = data.notna()
      groups[i] = data[notna_mask]
      invalid_count.append(len(data) - notna_mask.count())
    return invalid_count
  
  def get_groups_info(self, groups: list[pd.Series])->list[TableComparisonGroupInfoResource]:
    return list(map(
      lambda group: TableComparisonGroupInfoResource(
        name=str(group.name),
        invalid_size=group.isna().sum(),
        sample_size=len(group),
      ),
      groups
    ))
      
  def check_is_valid(self, groups: list[pd.Series])->StatisticTestValidityModel:
    validity1 = self._are_compared_groups_overlapping(groups)
    validity2 = self._are_samples_large_enough(groups)
    return validity1.merge(validity2)

  def preprocess(self, groups: list[pd.Series])->None:
    self._exclude_na_rows(groups)
    if self.exclude_overlapping_rows:
      self._exclude_overlapping_rows(groups)

  def load(self, df: pd.DataFrame, column: SchemaColumn)->list[pd.Series]:
    data_groups: list[pd.Series] = []
    for group in self.groups:
      filtered_df = self.engine.filter(df, group.filter)
      data = filtered_df[column.name]
      data.name = group.name
      data_groups.append(data)
    return data_groups
  
  def compare(self, df: pd.DataFrame, *, column_name: str, statistic_test_preference: StatisticTestMethodEnum = StatisticTestMethodEnum.Auto, effect_size_preference: EffectSizeMethodEnum = EffectSizeMethodEnum.Auto):
    column = self.config.data_schema.assert_exists(column_name)
    groups = self.load(df, column)
    self.preprocess(groups)
    validity = self.check_is_valid(groups)

    statistic_test_method = StatisticTestFactory(
      column=column,
      groups=groups,
      preference=statistic_test_preference
    ).build()

    validity = statistic_test_method.check_is_valid()
    significance = statistic_test_method.significance()

    effect_size_method = EffectSizeFactory(
      column=column,
      groups=groups,
      preference=effect_size_preference
    ).build(significance)
    
    validity2 = effect_size_method.check_is_valid()
    validity.merge(validity2)
    effect_size = effect_size_method.effect_size()

    group_info = self.get_groups_info(groups)

    return TableComparisonResource(
      effect_size=effect_size,
      significance=significance,
      groups=group_info,
      warnings=validity.warnings
    )

__all__ = [
  "TableComparisonEngine"
]