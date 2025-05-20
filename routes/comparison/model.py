
import pydantic
from modules.comparison.base import EffectSizeResult, SignificanceResult
from modules.comparison.effect_size import EffectSizeMethodEnum
from modules.comparison.engine import StatisticTestResult
from modules.comparison.statistic_test import GroupStatisticTestMethodEnum, StatisticTestMethodEnum
from modules.config.schema.schema_variants import SchemaColumn
from modules.table import NamedTableFilter
from modules.table.filter_variants import TableFilter

# Schema
class CompareSubdatasetsSchema(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  column: str

class SubdatasetCooccurrenceResource(pydantic.BaseModel):
  labels: list[str]
  cooccurrences: list[list[int]]
  frequencies: list[int]

