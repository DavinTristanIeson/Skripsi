import http
from typing import Any, Optional, cast

import pandas as pd
import numpy as np

from models.table import DescriptiveStatisticsResource
from modules.api import ApiResult
from modules.api.wrapper import ApiError
from modules.project.cache import get_cached_data_source
from modules.logger import ProvisionedLogger
from modules.config import SchemaColumnTypeEnum

from models.project import CheckDatasetColumnSchema, CheckDatasetResource, CheckDatasetSchema, InferDatasetColumnResource, DatasetPreviewResource

logger = ProvisionedLogger().provision("Project Controller")

def infer_column_by_type(column: str, df: pd.DataFrame, dtype: SchemaColumnTypeEnum):
  data = df[column]
  descriptive_statistics: Optional[DescriptiveStatisticsResource] = None
  categories: Optional[list[str]] = None
  if dtype == SchemaColumnTypeEnum.Textual:
    data = data.astype(str)
    descriptive_statistics = DescriptiveStatisticsResource.from_series(data.str.len())
  elif dtype == SchemaColumnTypeEnum.Continuous:
    data = data.astype(np.float64)
    descriptive_statistics = DescriptiveStatisticsResource.from_series(data)
  elif dtype == SchemaColumnTypeEnum.OrderedCategorical:
    data = data.astype(str)
    categories = sorted(set(map(str, data.unique())))

  logger.info(f"Inferred type {dtype} for column {column}.")
  return InferDatasetColumnResource(
    name=column,
    type=dtype,
    descriptive_statistics=descriptive_statistics,
    categories=categories,
    count=data.count(),
  )

def infer_column_without_type(column: str, df: pd.DataFrame)->InferDatasetColumnResource:
  data = df[column]
  dtype = data.dtype
  logger.info(f"Inferring autofill values for column {column}")
  if data.count() == 0:
    logger.warning(f"Column {column} is empty.")
    return InferDatasetColumnResource(
      name=column,
      type=SchemaColumnTypeEnum.Unique,
      descriptive_statistics=None,
      categories=None,
      count=0,
    )
  try:
    if pd.api.types.is_numeric_dtype(dtype):
      if column.lower().startswith("lat") or column.lower().startswith("long"):
        return infer_column_by_type(column, df, SchemaColumnTypeEnum.Geospatial)
      else:
        return infer_column_by_type(column, df, SchemaColumnTypeEnum.Continuous)
    else:
      uniquescnt = len(data.unique())
      if uniquescnt < 0.1 * data.count():
        return infer_column_by_type(column, df, SchemaColumnTypeEnum.Categorical)
      else:
        is_string = pd.api.types.is_string_dtype(dtype)
        has_long_text = is_string and data.str.len().mean() >= 20

        if has_long_text:
          return infer_column_by_type(column, df, SchemaColumnTypeEnum.Textual)
        else:
          return infer_column_by_type(column, df, SchemaColumnTypeEnum.Unique)
  except Exception as e:
    logger.error(e)
  logger.warning(f"Unable to infer any autofill values for column {column} as it doesn't fulfill any of the conditions.")
  return infer_column_by_type(column, df, SchemaColumnTypeEnum.Unique)


def infer_columns_from_dataset(body: CheckDatasetSchema):
  df = get_cached_data_source(body.root)
  columns: list[InferDatasetColumnResource] = []
  for column in df.columns:
    inferred = infer_column_without_type(column, df)
    columns.append(inferred)
  return ApiResult(
    data=CheckDatasetResource(
      columns=columns,
    ),
    message=f"We have inferred the columns from the dataset at {body.root.path}. Next, please configure how you would like to process the individual columns."
  )

def infer_column_from_dataset(body: CheckDatasetColumnSchema):
  df = get_cached_data_source(body.source)
  if len(set(df.columns)) != len(df.columns):
    raise ApiError("There are duplicate columns in the dataset!", http.HTTPStatus.BAD_REQUEST)
  inferred = infer_column_by_type(body.column, df, body.dtype)

  return ApiResult(
    data=inferred,
    message=None
  )

def get_dataset_preview(body: CheckDatasetSchema):
  df = get_cached_data_source(body.root)
  dataset_columns=list(df.columns)
  total_rows=len(df)

  df = df.head(15) 
  df["__index"] = df.index

  preview_rows=cast(list[dict[str, Any]], df.to_dict(orient="records"))
  return ApiResult(
    data=DatasetPreviewResource(
      dataset_columns=dataset_columns,
      total_rows=total_rows,
      preview_rows=preview_rows,
    ),
    message=None
  )

__all__ = [
  "infer_columns_from_dataset",
  "infer_column_from_dataset",
  "get_dataset_preview"
]