import functools
import http
from typing import Annotated, Callable, Optional, Sequence, cast

import pandas as pd
import pydantic

from common.logger import RegisteredLogger, TimeLogger
from common.models.api import ApiError
from .schema import CategoricalSchemaColumn, ContinuousSchemaColumn, GeospatialSchemaColumn, ImageSchemaColumn, SchemaColumn, SchemaColumnTypeEnum, TemporalSchemaColumn, TextualSchemaColumn, UniqueSchemaColumn

logger = RegisteredLogger().provision("Config")

def __validate_schema_manager_columns(value: list[SchemaColumn]):
    unique_names: set[str] = set()
    non_unique_names: set[str] = set()
    has_text_column = False

    for col in value:
      if col.type == SchemaColumnTypeEnum.Textual:
        has_text_column = True

      if col.name in unique_names:
        non_unique_names.add(col.name)
      else:
        unique_names.add(col.name)

    if len(non_unique_names) > 0:
      raise ValueError(f"All column names must be unique. Make sure that that there's only one of the following names: {', '.join(non_unique_names)}")
    if not has_text_column:
      raise ValueError(f"There should be at least one textual column in the dataset.")

    return list(filter(lambda x: x.active, value))

def __extend_schema_manager_columns(value: list[SchemaColumn]):
    offset = 0
    final_columns = list(value)
    for idx, col in enumerate(value):
      additional_internal_columns: Optional[list[SchemaColumn]] = None
      if col.type == SchemaColumnTypeEnum.Temporal:
        col = cast(TemporalSchemaColumn, col)
        additional_internal_columns = [
          col.year_column,
          col.month_column,
          col.day_of_week_column,
          col.hour_column,
        ]
      elif col.type == SchemaColumnTypeEnum.Textual:
        col = cast(TextualSchemaColumn, col)
        additional_internal_columns = [
          col.preprocess_column,
          col.topic_column,
        ]
      if additional_internal_columns is None:
        continue
      for col in additional_internal_columns:
        final_columns.insert(idx + offset, col)
        offset += 1
    return final_columns

def __serialize_columns(value: list[SchemaColumn], handler):
  return handler(list(filter(lambda x: not x.internal, value)))
  
  

SchemaColumnListField = Annotated[
  list[SchemaColumn],
  pydantic.AfterValidator(__validate_schema_manager_columns),
  pydantic.AfterValidator(__extend_schema_manager_columns),
  pydantic.WrapSerializer(__serialize_columns)
]
class SchemaManager(pydantic.BaseModel):
  columns: SchemaColumnListField

  def as_dictionary(self)->dict[str, SchemaColumn]:
    return {col.name: col for col in self.columns}

  def of_type(self, type: SchemaColumnTypeEnum)->list[SchemaColumn]:
    return list(filter(lambda x: x.type == type and x.active, self.columns))
  
  def textual(self)->list[TextualSchemaColumn]:
    return cast(list[TextualSchemaColumn], self.of_type(SchemaColumnTypeEnum.Textual))
  
  def unique(self)->list[UniqueSchemaColumn]:
    return cast(list[UniqueSchemaColumn], self.of_type(SchemaColumnTypeEnum.Unique))
  
  def continuous(self)->list[ContinuousSchemaColumn]:
    return cast(list[ContinuousSchemaColumn], self.of_type(SchemaColumnTypeEnum.Continuous))
  
  def categorical(self)->list[CategoricalSchemaColumn]:
    return cast(list[CategoricalSchemaColumn], self.of_type(SchemaColumnTypeEnum.Categorical))
  
  def temporal(self)->list[TemporalSchemaColumn]:
    return cast(list[TemporalSchemaColumn], self.of_type(SchemaColumnTypeEnum.Temporal))
  
  def image(self)->list[ImageSchemaColumn]:
    return cast(list[ImageSchemaColumn], self.of_type(SchemaColumnTypeEnum.Image))
  
  def geospatial(self)->list[GeospatialSchemaColumn]:
    return cast(list[GeospatialSchemaColumn], self.of_type(SchemaColumnTypeEnum.Geospatial))

  def assert_exists(self, name:str)->SchemaColumn:
    column: Optional[SchemaColumn] = None
    for col in self.columns:
      if col.name == name:
        column = col
    if column is None:
      raise ApiError(f"Column {name} doesn't exist in the schema", http.HTTPStatus.NOT_FOUND)
    return column
  
  def fit(self, df: pd.DataFrame)->pd.DataFrame:
    try:
      fitted_df = df.loc[:, [col.name for col in self.columns if col.active and not col.internal]]
    except ApiError as e:
      raise ApiError(f"The columns specified in the project configuration does not match the column in the dataset. Please remove this column from the project configuration if they do not exist in the dataset: {str(e)}", http.HTTPStatus.NOT_FOUND)
    
    for col in self.columns:
      if col.internal:
        continue
      with TimeLogger(logger, f"Fitting {col.name} ({col.type})"):
        col.fit(fitted_df)
    return fitted_df
    
    