from typing import cast

import pydantic
from modules.comparison.effect_size import EffectSizeMethodEnum
from modules.comparison.statistic_test import StatisticTestMethodEnum
from modules.config.config import Config
from modules.config.schema.base import SchemaColumnTypeEnum
from modules.config.schema.schema_variants import SchemaColumn, TextualSchemaColumn

# Schema
class TopicCorrelationSchema(pydantic.BaseModel):
  column1: str
  column2: str

  def topic_column(self, config: Config):
    raw_textual_column = config.data_schema.assert_of_type(self.column1, [SchemaColumnTypeEnum.Textual])
    textual_column = cast(TextualSchemaColumn, raw_textual_column)
    return textual_column.topic_column

# Resource
class StatisticTestOnDistributionResource(pydantic.BaseModel):
  statistic_test_method: StatisticTestMethodEnum
  effect_size_method: EffectSizeMethodEnum
  p_values: list[float]
  statistics: list[float]
  effect_sizes: list[float]

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

class FineGrainedStatisticTestOnCategoriesResource(pydantic.BaseModel):
  statistic_test_method: StatisticTestMethodEnum
  effect_size_method: EffectSizeMethodEnum  
  p_values: list[list[float]]
  statistics: list[list[float]]
  effect_sizes: list[list[float]]
