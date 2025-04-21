from typing import Any, cast

import pydantic
from modules.comparison.base import EffectSizeResult, SignificanceResult
from modules.comparison.effect_size import EffectSizeMethodEnum
from modules.comparison.engine import TableComparisonResult
from modules.comparison.statistic_test import StatisticTestMethodEnum
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

# Resource
class BinaryStatisticTestOnDistributionResource(pydantic.BaseModel):
  discriminator: str

  yes_count: int
  no_count: int
  invalid_count: int
  
  warnings: list[str]
  significance: SignificanceResult
  effect_size: EffectSizeResult

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
  statistic_test_method: StatisticTestMethodEnum
  effect_size_method: EffectSizeMethodEnum  
  p_values: list[list[float]]
  statistics: list[list[float]]
  effect_sizes: list[list[float]]
