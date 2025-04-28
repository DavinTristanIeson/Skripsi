from typing import Any, cast

import pydantic
from modules.comparison.base import EffectSizeResult, SignificanceResult
from modules.comparison.effect_size import EffectSizeMethodEnum
from modules.comparison.engine import TableComparisonResult
from modules.comparison.statistic_test import GroupStatisticTestMethodEnum, StatisticTestMethodEnum
from modules.config.config import Config
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import SchemaColumn, TextualSchemaColumn

# Schema
class TopicCorrelationSchema(pydantic.BaseModel):
  column1: str
  column2: str

class BinaryStatisticTestSchema(TopicCorrelationSchema, pydantic.BaseModel):
  statistic_test_preference: StatisticTestMethodEnum
  effect_size_preference: EffectSizeMethodEnum
  main_statistic_test_preference: GroupStatisticTestMethodEnum

# Resource
class BinaryStatisticTestOnDistributionResource(pydantic.BaseModel):
  discriminator: str

  yes_count: int
  no_count: int
  invalid_count: int
  
  warnings: list[str]
  significance: SignificanceResult
  effect_size: EffectSizeResult

class BinaryStatisticTestOnDistributionMainResource(pydantic.BaseModel):
  discriminators: list[str]
  discriminator_column: SchemaColumn
  target_column: SchemaColumn
  results: list[BinaryStatisticTestOnDistributionResource]
  significance: SignificanceResult
  effect_size: EffectSizeResult
  warnings: list[str]

class ContingencyTableResource(pydantic.BaseModel):
  column1: SchemaColumn
  column2: SchemaColumn
  rows: list[str]
  columns: list[str]
  observed: list[list[int]]
  expected: list[list[float]]
  residuals: list[list[float]]
  # Standardized residuals.
  standardized_residuals: list[list[float]]

class BinaryStatisticTestOnContingencyTableResource(pydantic.BaseModel):
  discriminator1: str
  discriminator2: str
  frequency: int
  
  warnings: list[str]
  significance: SignificanceResult
  effect_size: EffectSizeResult

class BinaryStatisticTestOnContingencyTableMainResource(pydantic.BaseModel):
  rows: list[str] 
  columns: list[str]
  column1: SchemaColumn
  column2: SchemaColumn
  results: list[list[BinaryStatisticTestOnContingencyTableResource]]
  warnings: list[str]
  significance: SignificanceResult
  effect_size: EffectSizeResult
