
import pydantic
from modules.comparison.effect_size import EffectSizeMethodEnum
from modules.comparison.statistic_test import StatisticTestMethodEnum
from modules.table import NamedTableFilter

class ComparisonStatisticTestSchema(pydantic.BaseModel):
  group1: NamedTableFilter
  group2: NamedTableFilter
  column: str

  statistic_test_preference: StatisticTestMethodEnum
  effect_size_preference: EffectSizeMethodEnum
  exclude_overlapping_rows: bool

class CompareSubdatasetsSchema(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  column: str

class SubdatasetCooccurrenceResource(pydantic.BaseModel):
  labels: list[str]
  cooccurrences: list[list[int]]
  frequencies: list[int]