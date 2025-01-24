from typing import Annotated, Optional

from fastapi import Depends, Query
import numpy as np
import pandas as pd
from common.logger import RegisteredLogger
from common.models.api import ApiError

from models.config import Config, SchemaColumn, SchemaColumnTypeEnum, DocumentEmbeddingMethodEnum
from models.project import InferDatasetColumnResource, InferDatasetCategoryFrequenciesResource, InferDatasetDescriptiveStatisticsResource

def get_project_config(project_id: str):
  return Config.from_project(project_id)

ProjectExistsDependency = Annotated[Config, Depends(get_project_config)]

def get_data_column(config: ProjectExistsDependency, column: str = Query()):
  try:
    return config.data_schema.assert_exists(column)
  except KeyError:
    raise ApiError(f"Column {column} doesn't exist in the schema. Please make sure that your schema is properly configured to your data.", 404)
SchemaColumnExistsDependency = Annotated[SchemaColumn, Depends(get_data_column)]

logger = RegisteredLogger().provision("Project Controller")

def infer_column(column: str, df: pd.DataFrame)->InferDatasetColumnResource:
  data = df[column]
  dtype = data.dtype
  coltype: SchemaColumnTypeEnum
  descriptive_statistics: Optional[InferDatasetDescriptiveStatisticsResource] = None
  category_frequencies: Optional[InferDatasetCategoryFrequenciesResource] = None
  if pd.api.types.is_numeric_dtype(dtype):
    coltype = SchemaColumnTypeEnum.Continuous
    descriptive_statistics = InferDatasetDescriptiveStatisticsResource.from_series(data)
  else:
    uniquescnt = len(data.unique())
    if uniquescnt < 0.2 * len(column):
      coltype = SchemaColumnTypeEnum.Categorical
      category_frequencies = InferDatasetCategoryFrequenciesResource.from_categorical_series(data)
    else:
      is_string = pd.api.types.is_string_dtype(dtype)
      has_long_text = is_string and data.str.len().mean() >= 20

      if has_long_text:
        is_json_column = (data.str.startswith('[') & data.str.endswith(']')).count() == data.count()
        if is_json_column:
          coltype = SchemaColumnTypeEnum.MultiCategorical
          try:
            category_frequencies = InferDatasetCategoryFrequenciesResource.from_multi_categorical_series(data)
          except Exception as e:
            logger.error(e)
            coltype = SchemaColumnTypeEnum.Textual
        else:
          coltype = SchemaColumnTypeEnum.Textual
      else:
        coltype = SchemaColumnTypeEnum.Unique

  return InferDatasetColumnResource(
    name=column,
    type=coltype,
    descriptive_statistics=descriptive_statistics,
    category_frequencies=category_frequencies,
    total_rows=len(data),
  )


__all__ = [
  "ProjectExistsDependency",
  "SchemaColumnExistsDependency",
  "infer_column"
]