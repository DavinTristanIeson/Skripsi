from typing import Optional

import pandas as pd

from modules.api import ApiResult
from modules.config.cache import get_cached_data_source
from modules.logger import ProvisionedLogger
from modules.config import SchemaColumnTypeEnum

from models.project import CheckDatasetColumnSchema, CheckDatasetResource, CheckDatasetSchema, InferDatasetColumnResource, InferDatasetDescriptiveStatisticsResource

logger = ProvisionedLogger().provision("Project Controller")

def infer_column_by_type(column: str, df: pd.DataFrame, dtype: SchemaColumnTypeEnum):
  data = df[column]
  document_lengths: Optional[InferDatasetDescriptiveStatisticsResource] = None
  categories: Optional[list[str]] = None
  if dtype == SchemaColumnTypeEnum.Textual:
    data = data.astype(str)
    document_lengths = InferDatasetDescriptiveStatisticsResource.from_series(data.str.len())
  elif dtype == SchemaColumnTypeEnum.OrderedCategorical:
    data = data.astype(str)
    categories = sorted(set(map(str, data.unique())))

  logger.info(f"Inferred type {dtype} for column {column}.")
  return InferDatasetColumnResource(
    name=column,
    type=dtype,
    document_lengths=document_lengths,
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
      document_lengths=None,
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
      dataset_columns=list(df.columns),
      total_rows=len(df),
      preview_rows=df.head(5).to_dict(orient="records")
    ),
    message=f"We have inferred the columns from the dataset at {body.root.path}. Next, please configure how you would like to process the individual columns."
  )

def infer_column_from_dataset(body: CheckDatasetColumnSchema):
  df = get_cached_data_source(body.source)
  inferred = infer_column_by_type(body.column, df, body.dtype)

  return ApiResult(
    data=inferred,
    message=None
  )

__all__ = [
  "infer_columns_from_dataset",
  "infer_column_from_dataset",
]