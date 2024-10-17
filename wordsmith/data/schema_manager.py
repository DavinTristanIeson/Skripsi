from typing import Sequence, cast

import pandas as pd
import pydantic

from common.logger import RegisteredLogger, TimeLogger
from wordsmith.data.schema import CategoricalSchemaColumn, ContinuousSchemaColumn, SchemaColumn, SchemaColumnType, TemporalSchemaColumn, TextualSchemaColumn, UniqueSchemaColumn

logger = RegisteredLogger().provision("Config")
class SchemaManager(pydantic.BaseModel):
  columns: Sequence[SchemaColumn]

  def of_type(self, type: SchemaColumnType)->tuple[SchemaColumn, ...]:
    return tuple(filter(lambda x: x.type == type, self.columns))
  
  def textual(self)->tuple[TextualSchemaColumn, ...]:
    return cast(tuple[TextualSchemaColumn, ...], self.of_type(SchemaColumnType.Textual))
  
  def unique(self)->tuple[UniqueSchemaColumn, ...]:
    return cast(tuple[UniqueSchemaColumn, ...], self.of_type(SchemaColumnType.Unique))
  
  def continuous(self)->tuple[ContinuousSchemaColumn, ...]:
    return cast(tuple[ContinuousSchemaColumn, ...], self.of_type(SchemaColumnType.Continuous))
  
  def categorical(self)->tuple[CategoricalSchemaColumn, ...]:
    return cast(tuple[CategoricalSchemaColumn, ...], self.of_type(SchemaColumnType.Categorical))
  
  def temporal(self)->tuple[TemporalSchemaColumn, ...]:
    return cast(tuple[TemporalSchemaColumn, ...], self.of_type(SchemaColumnType.Temporal))
  
  def preprocess(self, df: pd.DataFrame):
    df = df.loc[:, [col.name for col in self.columns]]
    
    for col in self.columns:
      df.loc[:, col.name] = col.fit(cast(pd.Series, df.loc[:, col.name]))
      if col.type != SchemaColumnType.Textual:
        continue

      with TimeLogger(logger, f"Preprocessing {col.name}"):
        df[col.preprocess_column] = pd.Series(col.preprocessing.apply(df[col.name], show_progress=show_progress)) # type: ignore

    return df
