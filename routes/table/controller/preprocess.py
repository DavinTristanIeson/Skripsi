
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional, cast
import pandas as pd

from modules.api.wrapper import ApiError
from modules.config.schema.base import GeospatialRoleEnum, SchemaColumnTypeEnum
from modules.config.schema.schema_variants import GeospatialSchemaColumn, SchemaColumn
from modules.project.cache import ProjectCache
from modules.table.engine import TableEngine
from modules.table.filter import TableSort
from modules.table.filter_variants import TableFilter


@dataclass
class TableFilterPreprocessResult:
  df: pd.DataFrame
  data: pd.Series
  column: SchemaColumn

@dataclass
class TableFilterGeographicalPreprocessResult:
  df: pd.DataFrame
  latitude_column: SchemaColumn
  longitude_column: SchemaColumn
  latitudes: list[float]
  longitudes: list[float]

@dataclass
class TablePreprocessModule:
  cache: ProjectCache

  def get_data(self, df: pd.DataFrame, column: SchemaColumn, *, exclude_invalid=True, transform_data: bool = True):
    if column.name not in df.columns:
      raise ApiError(f"The column \"{column.name}\" does not exist in the dataset. There may have been some sort of data corruption in the application.", HTTPStatus.NOT_FOUND)
    data = df[column.name]

    if exclude_invalid:
      mask = data.notna()
      if column.type == SchemaColumnTypeEnum.Topic:
        mask = mask & (data != -1)
      data = data[mask]
      df = df[mask]
    
    if len(df) == 0:
      raise ApiError("There are no rows that can be visualized. Perhaps the filter is too strict; try adjusting the filter to be more lax.", HTTPStatus.BAD_REQUEST)

    # Use categorical dtype for topic
    if transform_data:
      if column.type == SchemaColumnTypeEnum.Topic:
        tm_result = self.cache.topics.load(cast(str, column.source_name))
        categorical_data = pd.Categorical(data)
        categorical_data = categorical_data.rename_categories(tm_result.renamer)
        data = pd.Series(categorical_data, name=column.name)
      if column.type == SchemaColumnTypeEnum.Boolean:
        categorical_data = pd.Categorical(data)
        categorical_data = categorical_data.rename_categories({
          True: "True",
          False: "False"
        })
        data = pd.Series(categorical_data, name=column.name)
    return data
  
  def assert_column(self, column_name: str, supported_types: Optional[list[SchemaColumnTypeEnum]] = None):
    config = self.cache.config
    if supported_types is not None:
      column = config.data_schema.assert_of_type(column_name, supported_types)
    else:
      column = config.data_schema.assert_exists(column_name)
    return column
    
  def load_dataframe(self, column: SchemaColumn, filter: Optional[TableFilter])->pd.DataFrame:
    engine = TableEngine(config=self.cache.config)

    # Sort the results first for ordered categorical and temporal
    sort: Optional[TableSort] = None
    if sort is None:
      if column.is_ordered:
        sort = TableSort(name=column.name, asc=True)
      elif column.type == SchemaColumnTypeEnum.Boolean:
        # True before False
        sort = TableSort(name=column.name, asc=False)

    df = engine.process_workspace(filter, sort)
    if len(df) == 0:
      raise ApiError(
        message=f"There are no valid rows in the dataset after the filter has been applied. Perhaps your filters are too strict?",
        status_code=HTTPStatus.BAD_REQUEST,
      )

    return df
  
  def apply(self, column_name: str, filter: Optional[TableFilter], *, supported_types: Optional[list[SchemaColumnTypeEnum]] = None, exclude_invalid=True, transform_data: bool = True):
    column = self.assert_column(column_name, supported_types)
    df = self.load_dataframe(column, filter)
    data = self.get_data(df, column, exclude_invalid=exclude_invalid, transform_data=transform_data)
    return TableFilterPreprocessResult(
      column=column,
      df=df,
      data=data,
    )

  def apply_geographical(self, filter: Optional[TableFilter], *, latitude_column_name: str, longitude_column_name: str, additional_column_constraints: dict[str, list[SchemaColumnTypeEnum]], aggregator: dict[str, pd.NamedAgg]):
    df = self.cache.workspaces.load()
    config = self.cache.config

    latitude_column = cast(GeospatialSchemaColumn, config.data_schema.assert_of_type(latitude_column_name, [SchemaColumnTypeEnum.Geospatial]))
    longitude_column = cast(GeospatialSchemaColumn, config.data_schema.assert_of_type(longitude_column_name, [SchemaColumnTypeEnum.Geospatial]))

    additional_columns = []
    for (column_name, column_type_constraints) in additional_column_constraints.items():
      additional_column = config.data_schema.assert_of_type(column_name, column_type_constraints)
      additional_columns.append(additional_column)

    if latitude_column.role != GeospatialRoleEnum.Latitude:
      raise ApiError(f"\"{latitude_column}\" is a column of type \"Geospatial\", but it does not contain latitude values. Perhaps you meant to use this column as a longitude column?", HTTPStatus.UNPROCESSABLE_ENTITY)
    if longitude_column.role != GeospatialRoleEnum.Longitude:
      raise ApiError(f"\"{latitude_column}\" is a column of type \"Geospatial\", but it does not contain longitude values. Perhaps you meant to use this column as a latitude column?", HTTPStatus.UNPROCESSABLE_ENTITY)

    engine = TableEngine(config=config)
    filtered_df = engine.filter(df, filter)

    check_these_columns = [latitude_column.name, longitude_column.name]
    check_these_columns.extend(additional_column_constraints.keys())
    for column in check_these_columns:
      if column not in filtered_df.columns:
        raise ApiError(f"The column \"{column}\" does not exist in the dataset. There may have been some sort of data corruption in the application.", HTTPStatus.NOT_FOUND)

    # Remove NA and invalid values
    latitude_raw = filtered_df[latitude_column.name]
    latitude_mask = latitude_raw.notna()
    longitude_raw = filtered_df[longitude_column.name]
    longitude_mask = longitude_raw.notna()

    coordinate_mask = latitude_mask & longitude_mask
    coordinates = filtered_df[coordinate_mask]

    # Count duplicates
    # https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe
    unique_coordinates = (coordinates
      .groupby([latitude_column.name, longitude_column.name], as_index=False)
      .agg(**aggregator)) # type: ignore
        
    latitudes = unique_coordinates.loc[:, latitude_column.name].to_list()
    longitudes = unique_coordinates.loc[:, longitude_column.name].to_list()
    
    return TableFilterGeographicalPreprocessResult(
      latitude_column=latitude_column,
      longitude_column=longitude_column,
      latitudes=latitudes,
      longitudes=longitudes,
      df=unique_coordinates
    )
