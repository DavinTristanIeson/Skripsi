import pydantic

from modules.table import NamedTableFilter

from modules.comparison import EffectSizeMethodEnum, StatisticTestMethodEnum

# Schema
class TableComparisonSchema(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  statistic_test_method: StatisticTestMethodEnum
  effect_size_method: EffectSizeMethodEnum
  exclude_overlapping_rows: bool


__all__ = [
  "TableComparisonSchema"
]