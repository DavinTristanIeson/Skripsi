# Schema
import pydantic
from modules.comparison.base import EffectSizeResult, SignificanceResult
from modules.comparison.effect_size import EffectSizeMethodEnum
from modules.comparison.engine import StatisticTestResult
from modules.comparison.statistic_test import OmnibusStatisticTestMethodEnum, StatisticTestMethodEnum
from modules.config.schema.schema_variants import SchemaColumn
from modules.table.filter_variants import NamedTableFilter


class StatisticTestSchema(pydantic.BaseModel):
  group1: NamedTableFilter
  group2: NamedTableFilter
  column: str

  statistic_test_preference: StatisticTestMethodEnum
  effect_size_preference: EffectSizeMethodEnum

class PairwiseStatisticTestSchema(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  column: str

  statistic_test_preference: StatisticTestMethodEnum
  effect_size_preference: EffectSizeMethodEnum

class BinaryStatisticTestSchema(pydantic.BaseModel):
  column: str
  groups: list[NamedTableFilter]
  statistic_test_preference: StatisticTestMethodEnum
  effect_size_preference: EffectSizeMethodEnum

class OmnibusStatisticTestSchema(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  column: str
  statistic_test_preference: OmnibusStatisticTestMethodEnum

class GetContingencyTableSchema(pydantic.BaseModel):
  groups: list[NamedTableFilter]
  column: str
  

# Resource

class ContingencyTableResource(pydantic.BaseModel):
  column: SchemaColumn
  rows: list[str]
  columns: list[str]
  observed: list[list[int]]
  expected: list[list[float]]
  residuals: list[list[float]]
  # Standardized residuals.
  standardized_residuals: list[list[float]]

class PairwiseStatisticTestResultResource(pydantic.BaseModel):
  column: SchemaColumn
  groups: list[str]
  results: list[StatisticTestResult]

class BinaryStatisticTestOnDistributionResultResource(pydantic.BaseModel):
  column: SchemaColumn
  groups: list[str]
  results: list[StatisticTestResult]

class BinaryStatisticTestOnContingencyTableResultResource(pydantic.BaseModel):
  discriminator1: str
  discriminator2: str
  frequency: int
  
  warnings: list[str]
  significance: SignificanceResult
  effect_size: EffectSizeResult

class BinaryStatisticTestOnContingencyTableResultMainResource(pydantic.BaseModel):
  column: SchemaColumn
  rows: list[str] 
  columns: list[str]
  results: list[list[BinaryStatisticTestOnContingencyTableResultResource]]
  