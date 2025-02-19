
import pydantic

from models.comparison.base import EffectSizeResult, NamedTableFilter, SignificanceResult
from models.comparison.effect_size import EffectSizeMethodEnum
from models.comparison.statistic_test import StatisticTestMethodEnum

# Schema
class TableComparisonSchema(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  statistic_test_method: StatisticTestMethodEnum
  effect_size_method: EffectSizeMethodEnum
  exclude_overlapping_rows: bool

# Resource

class TableComparisonGroupInfoResource(pydantic.BaseModel):
  name: str
  sample_size: int
  invalid_size: int

class TableComparisonResource(pydantic.BaseModel):
  warnings: list[str]
  groups: list[TableComparisonGroupInfoResource]
  significance: SignificanceResult
  effect_size: EffectSizeResult

__all__ = [
  "TableComparisonGroupInfoResource",
  "TableComparisonResource",
  "TableComparisonSchema"
]