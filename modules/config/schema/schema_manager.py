from dataclasses import dataclass
import http
from typing import Annotated, Optional, Sequence, cast

import pandas as pd
import pydantic

from modules.config.context import ConfigSerializationContext
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
    additional_internal_columns = col.get_internal_columns()
    if additional_internal_columns is None or len(additional_internal_columns) == 0:
      continue
    for col in additional_internal_columns:
      final_columns.insert(idx + offset + 1, cast(SchemaColumn, col))
      offset += 1
  return final_columns

def __serialize_columns(value: list[SchemaColumn], handler, info: pydantic.SerializationInfo):
  if isinstance(info.context, ConfigSerializationContext) and info.context.is_save:
    return handler(list(filter(lambda x: not x.internal, value)))
  return handler(value)
  
SchemaColumnListField = Annotated[
  list[SchemaColumn],
  pydantic.AfterValidator(__validate_schema_manager_columns),
  pydantic.AfterValidator(__extend_schema_manager_columns),
  pydantic.WrapSerializer(__serialize_columns)
]

@dataclass
class _SchemaColumnDiff:
  previous: SchemaColumn
  current: Optional[SchemaColumn]

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

  def non_internal(self)->list[SchemaColumn]:
    return list(filter(lambda col: not col.internal, self.columns))

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

  def _ensure_columns_in_df(self, raw_df: pd.DataFrame):
    try:
      column_targets = list(map(lambda col: col.name, self.non_internal()))
      df = raw_df.loc[:, column_targets]
      return df
    except (IndexError, KeyError) as e:
      logger.error(e)
      raise ApiError(f"The columns specified in the project configuration does not match the column in the dataset. Please remove this column from the project configuration if they do not exist in the dataset: {e.args}", http.HTTPStatus.NOT_FOUND)

  def fit(self, raw_df: pd.DataFrame)->pd.DataFrame:
    df = self._ensure_columns_in_df(raw_df)
    logger.debug(f"Fitting dataframe with columns: {raw_df.columns} with the following columns {list(map(lambda col: col.name, self.columns))}")
    for col in self.non_internal():
      self.__fit_column(df, col)
    return df
  
  def resolve_column_difference(self, previous: Sequence[SchemaColumn])->list[_SchemaColumnDiff]:  
    previous_column_map = {col.name: col for col in previous if not col.internal}
    non_internal_columns = filter(lambda col: not col.internal, self.columns)
    different_columns = filter(lambda col: col != previous_column_map[col.name], non_internal_columns)
    
    column_diffs: list[_SchemaColumnDiff] = []

    # Find all the columns with different configurations
    for col in different_columns:
      column_diffs.append(_SchemaColumnDiff(
        previous=previous_column_map[col.name],
        current=col
      ))
    return column_diffs
    
  def resolve_difference(self, prev: "SchemaManager", workspace_df: pd.DataFrame, source_df: pd.DataFrame)->tuple[pd.DataFrame, list[_SchemaColumnDiff]]:
    # Check if dataset has changed.
    different_row_counts = len(source_df) != len(workspace_df)
    self_columns = self.non_internal()
    prev_columns = prev.non_internal()
    different_column_counts = len(set(map(lambda col: col.name, prev_columns)).symmetric_difference(map(lambda col: col.name, self_columns))) > 0

    if different_row_counts or different_column_counts:
      # Dataset has DEFINITELY changed. Refit everything.
      df = self.fit(source_df)
      column_diffs = list(map(lambda col: _SchemaColumnDiff(previous=col, current=None), self_columns)) # type: ignore
      return df, column_diffs
    
    source_df = self._ensure_columns_in_df(source_df)
    df = self._ensure_columns_in_df(workspace_df)

    # Invariants:
    # 1. There are no new columns added/removed.
    # 2. Source DF and Workspace DF is guaranteed to have the same columns.
    # 3. Source DF and Workspace DF have the same number of rows

    # The only difference between the new schema and the old schema is the types and configuration.

    # columns should be frozen classes for this to work.
    column_diffs = self.resolve_column_difference(prev_columns)

    # First loop: Clean up previous data
    for diff in column_diffs:
      if diff.current is None or diff.previous.type != diff.current.type:
        # Clean up all previous internal columns if type changed.
        internal_columns = diff.previous.get_internal_columns()
        for internal_column in internal_columns:
          try:
            # We can guarantee that ``df`` is only used here. This inplace is safe.
            df.drop(
              internal_column.name,
              axis=1, inplace=True
            )
          except KeyError:
            continue

    # Second loop: re-fit columns that actually changed.
    for diff in column_diffs:
      if diff.current is None:
        continue
      logger.info(f"Fitting \"{diff.current.name}\" again since the column options has changed.")
      # Fit using the source column
      # This is safe to do due to invariant (2) and (3).

      # Since __fit_column mutates df, we need to copy the source data to the column first.
      df[diff.current.name] = source_df[diff.current.name]
      self.__fit_column(df, diff.current)
      
    return df, column_diffs
    
__all__ = [
  "SchemaManager",
]