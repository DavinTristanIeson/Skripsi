from typing import Callable, Optional, Sequence, cast

import pandas as pd
import pydantic

from common.logger import RegisteredLogger, TimeLogger
from common.models.api import ApiError
from .schema import CategoricalSchemaColumn, ContinuousSchemaColumn, GeospatialSchemaColumn, ImageSchemaColumn, SchemaColumn, SchemaColumnTypeEnum, TemporalSchemaColumn, TextualSchemaColumn, UniqueSchemaColumn

logger = RegisteredLogger().provision("Config")
class SchemaManager(pydantic.BaseModel):
  columns: list[SchemaColumn]

  @pydantic.field_validator("columns", mode="after")
  def __validate_columns(cls, value: list[SchemaColumn]):
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
      raise KeyError(f"Column {name} doesn't exist in the schema")
    return column
  
  def fit(self, df: pd.DataFrame)->pd.DataFrame:
    try:
      fitted_df = df.loc[:, [col.name for col in self.columns if col.active]]
    except KeyError as e:
      raise KeyError(f"The columns specified in the project configuration does not match the column in the dataset. Please remove this column from the project configuration if they do not exist in the dataset: {str(e)}")
    
    for col in self.columns:
      with TimeLogger(logger, f"Fitting {col.name} ({col.type})"):
        fitted_df[col.name] = col.fit(fitted_df[col.name])
    return fitted_df
    

