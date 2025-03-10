from dataclasses import dataclass
import http
from typing import Annotated, Callable, Optional, Sequence, cast

import pandas as pd
import pydantic

from modules.logger import ProvisionedLogger, TimeLogger
from modules.api import ApiError
from .schema_variants import CategoricalSchemaColumn, OrderedCategoricalSchemaColumn, ContinuousSchemaColumn, GeospatialSchemaColumn, SchemaColumn, SchemaColumnTypeEnum, TemporalSchemaColumn, TextualSchemaColumn, TopicSchemaColumn, UniqueSchemaColumn

logger = ProvisionedLogger().provision("Config")

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

  return list(value)

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

@dataclass
class _SchemaColumnDiff:
  previous: SchemaColumn
  current: SchemaColumn

class SchemaManager(pydantic.BaseModel):
  columns: SchemaColumnListField

  def as_dictionary(self)->dict[str, SchemaColumn]:
    return {col.name: col for col in self.columns}

  def of_type(self, type: SchemaColumnTypeEnum)->list[SchemaColumn]:
    return list(filter(lambda x: x.type == type, self.columns))
  
  def textual(self)->list[TextualSchemaColumn]:
    return cast(list[TextualSchemaColumn], self.of_type(SchemaColumnTypeEnum.Textual))
  
  def unique(self)->list[UniqueSchemaColumn]:
    return cast(list[UniqueSchemaColumn], self.of_type(SchemaColumnTypeEnum.Unique))
  
  def continuous(self)->list[ContinuousSchemaColumn]:
    return cast(list[ContinuousSchemaColumn], self.of_type(SchemaColumnTypeEnum.Continuous))
  
  def categorical(self)->list[CategoricalSchemaColumn]:
    return cast(list[CategoricalSchemaColumn], self.of_type(SchemaColumnTypeEnum.Categorical))
  
  def ordered_categorical(self)->list[OrderedCategoricalSchemaColumn,]:
    return cast(list[OrderedCategoricalSchemaColumn], self.of_type(SchemaColumnTypeEnum.OrderedCategorical))
  
  def temporal(self)->list[TemporalSchemaColumn]:
    return cast(list[TemporalSchemaColumn], self.of_type(SchemaColumnTypeEnum.Temporal))

  def geospatial(self)->list[GeospatialSchemaColumn]:
    return cast(list[GeospatialSchemaColumn], self.of_type(SchemaColumnTypeEnum.Geospatial))
  
  def topic(self)->list[TopicSchemaColumn]:
    return cast(list[TopicSchemaColumn], self.of_type(SchemaColumnTypeEnum.Topic))

  def assert_exists(self, name:str)->SchemaColumn:
    column: Optional[SchemaColumn] = None
    for col in self.columns:
      if col.name == name:
        column = col
    if column is None:
      raise ApiError(f"Column \"{name}\" doesn't exist in the schema", http.HTTPStatus.NOT_FOUND)
    return column
  
  def assert_of_type(self, name: str, types: list[SchemaColumnTypeEnum])->SchemaColumn:
    column = self.assert_exists(name)
    if column.type not in types:
      raise ApiError(f"This procedure requires that column \"{name}\" be one of these types: {', '.join(types)}", http.HTTPStatus.UNPROCESSABLE_ENTITY)
    return column
  
  def process_columns(self, df: pd.DataFrame):
    reordered_df = df.loc[:, [col.name for col in self.columns if col.name in df.columns]]
    reordered_df.columns = [col.alias or col.name for col in self.columns]
    return reordered_df

  def __fit_column(self, df: pd.DataFrame, col: SchemaColumn):
    if col.internal:
      return
    with TimeLogger(logger, f"Fitting {col.name} ({col.type})"):
      try:
        col.fit(df)
      except Exception as e:
        logger.error(e)
        raise ApiError(f"An error has occurred while fitting \"{col.name}\": {e}. Please check if your dataset matches your column configuration.", http.HTTPStatus.UNPROCESSABLE_ENTITY)

  def fit(self, raw_df: pd.DataFrame)->pd.DataFrame:
    try:
      # don't exclude inactive columns, this is required for more effective updating.
      df = raw_df.loc[:, [col.name for col in self.columns if not col.internal]]
    except (IndexError, KeyError) as e:
      logger.error(e)
      raise ApiError(f"The columns specified in the project configuration does not match the column in the dataset. Please remove this column from the project configuration if they do not exist in the dataset: {e.args}", http.HTTPStatus.NOT_FOUND)
    
    for col in self.columns:
      self.__fit_column(df, col)
    return df
  
  def resolve_column_difference(self, previous: Sequence[SchemaColumn])->list[_SchemaColumnDiff]:
    current = self.columns
    prev_columns = set(map(lambda x: x.name, previous))
    current_columns = set(map(lambda x: x.name, current))
    columnar_differences = prev_columns.symmetric_difference(current_columns)
    if len(columnar_differences) > 0:
      raise ApiError(f"There are columns that only exist in one of the configurations but not in both: {list(columnar_differences)}. To avoid data corruption, we do not allow users to modify the columns or the data source after the creation of a project. Consider creating a new project instead.", http.HTTPStatus.BAD_REQUEST)
  
    previous_column_map = {col.name: col for col in previous}
    non_internal_columns = filter(lambda col: not col.internal, self.columns)
    different_columns = filter(lambda col: col != previous_column_map[col.name], non_internal_columns)
    
    column_diffs: list[_SchemaColumnDiff] = []
    for col in different_columns:
      column_diffs.append(_SchemaColumnDiff(
        previous=previous_column_map[col.name],
        current=col
      ))
    return column_diffs
    
  def resolve_difference(self, prev: "SchemaManager", workspace_df: pd.DataFrame, source_df: pd.DataFrame)->tuple[pd.DataFrame, list[_SchemaColumnDiff]]:
    # Check if dataset has changed.
    if len(source_df) != len(workspace_df):
      # Dataset has DEFINITELY changed. Refit everything.
      try:
        df = self.fit(source_df)
        column_diffs = list(map(lambda col: _SchemaColumnDiff(previous=col, current=None), self.columns)) # type: ignore
        return df, column_diffs
      except ApiError as e:
        raise ApiError(f"{e.message}. This problem may be caused by changes to the columns of the source dataset. In which case, please create a new project instead.", http.HTTPStatus.NOT_FOUND)

    column_diffs = self.resolve_column_difference(prev.columns)
    df = workspace_df.copy()
    for diff in column_diffs:
      internal_columns = diff.previous.get_internal_columns()
      if diff.previous.type != diff.current.type:
        df.drop(
          list(map(lambda x: x.name, internal_columns)),
          axis=1, inplace=True
        )
    for diff in column_diffs:
      logger.info(f"Fitting \"{diff.current.name}\" again since the column options has changed.")
      self.__fit_column(df, diff.current)
    return self.process_columns(df), column_diffs
    
__all__ = [
  "SchemaManager",
]