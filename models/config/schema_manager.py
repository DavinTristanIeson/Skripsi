from typing import Callable, Optional, Sequence, cast

import pandas as pd
import pydantic

from common.logger import RegisteredLogger, TimeLogger
from common.models.api import ApiError
from .schema import CategoricalSchemaColumn, ContinuousSchemaColumn, GeospatialSchemaColumn, SchemaColumn, SchemaColumnTypeEnum, TemporalSchemaColumn, TextualSchemaColumn, UniqueSchemaColumn

logger = RegisteredLogger().provision("Config")
class SchemaManager(pydantic.BaseModel):
  columns: Sequence[SchemaColumn]

  @pydantic.field_validator("columns", mode="after")
  def __validate_columns(cls, value: Sequence[SchemaColumn]):
    unique_names: set[str] = set()
    unique_dataset_names: set[str] = set()

    non_unique_names: set[str] = set()
    non_unique_dataset_names: set[str] = set()
    has_text_column = False
    for col in value:
      if col.type == SchemaColumnTypeEnum.Textual:
        has_text_column = True

      if col.name in unique_names:
        non_unique_names.add(col.name)
      else:
        unique_names.add(col.name)

      if col.type == SchemaColumnTypeEnum.Geospatial:
        geocol = cast(GeospatialSchemaColumn, col)
        if geocol.longitude_dataset_name not in value:
          raise ValueError(f'The column "{geocol.longitude_dataset_name}" which is the longitude part of "{geocol.name}" does not exist in the dataset. Please check the column configuration again and change the Longitude Dataset Name to the name of the longitude column (not the dataset name).')
        elif geocol.latitude_dataset_name not in value:
          raise ValueError(f'The column "{geocol.latitude_dataset_name}" which is the latitude part of "{geocol.name}" does not exist in the dataset. Please check the column configuration again and change the Latitude Dataset Name to the name of the latitude column (not the dataset name).')
        
        if geocol.longitude_dataset_name in unique_dataset_names:
          non_unique_dataset_names.add(geocol.longitude_dataset_name)
        else:
          unique_dataset_names.add(geocol.latitude_dataset_name)
          
        if geocol.latitude_dataset_name in unique_dataset_names:
          non_unique_dataset_names.add(geocol.latitude_dataset_name)
        else:
          unique_dataset_names.add(geocol.latitude_dataset_name)


      else:
        if col.dataset_name is None:
          continue

        if col.dataset_name in unique_dataset_names:
          non_unique_dataset_names.add(col.dataset_name)
        else:
          unique_dataset_names.add(col.dataset_name)


    if len(non_unique_names) > 0:
      raise ValueError(f"All column names must be unique. Make sure that that there's only one of the following names: {', '.join(non_unique_names)}")
    if len(non_unique_dataset_names) > 0:
      raise ValueError(f"All dataset column names must be unique. Make sure that that there's only one of the following names: {', '.join(non_unique_dataset_names)}")
    if not has_text_column:
      raise ValueError(f"There should be at least one textual column in the dataset.")

    return value
    
  def of_type(self, type: SchemaColumnTypeEnum)->tuple[SchemaColumn, ...]:
    return tuple(filter(lambda x: x.type == type, self.columns))
  
  def textual(self)->tuple[TextualSchemaColumn, ...]:
    return cast(tuple[TextualSchemaColumn, ...], self.of_type(SchemaColumnTypeEnum.Textual))
  
  def unique(self)->tuple[UniqueSchemaColumn, ...]:
    return cast(tuple[UniqueSchemaColumn, ...], self.of_type(SchemaColumnTypeEnum.Unique))
  
  def continuous(self)->tuple[ContinuousSchemaColumn, ...]:
    return cast(tuple[ContinuousSchemaColumn, ...], self.of_type(SchemaColumnTypeEnum.Continuous))
  
  def categorical(self)->tuple[CategoricalSchemaColumn, ...]:
    return cast(tuple[CategoricalSchemaColumn, ...], self.of_type(SchemaColumnTypeEnum.Categorical))
  
  def temporal(self)->tuple[TemporalSchemaColumn, ...]:
    return cast(tuple[TemporalSchemaColumn, ...], self.of_type(SchemaColumnTypeEnum.Temporal))
  
  def assert_exists(self, name:str)->SchemaColumn:
    column: Optional[SchemaColumn] = None
    for col in self.columns:
      if col.name == name:
        column = col
    if column is None:
      raise KeyError(f"Column {name} doesn't exist in the schema")
    return column
  
  def preprocess(self, df: pd.DataFrame):
    try:
      df = df.loc[:, [col.dataset_name or col.name for col in self.columns]]
    except KeyError as e:
      raise KeyError(f"The columns specified in the project configuration does not match the column in the dataset. Please remove this column from the project configuration if they do not exist in the dataset: {str(e)}")
    
    for col in self.columns:
      yield df, col
      with TimeLogger(logger, f"Preprocessing {col.name} ({col.type})"):
        col.fit(df)

