import abc
from enum import Enum
from typing import Annotated, ClassVar, Literal, Optional, Union, cast
import pydantic
import pandas as pd

from .preprocessing import PreprocessingConfig

class SchemaColumnType(str, Enum):
  Range = "range"
  Categorical = "categorical"
  Text = "text"
  Unique = "unique"

class BaseSchemaColumn(abc.ABC):
  name: str
  @abc.abstractmethod
  def fit(self, data: pd.Series)->pd.Series:
    pass

class RangeSchemaColumn(pydantic.BaseModel, BaseSchemaColumn):
  name: str
  type: Literal[SchemaColumnType.Range]
  # Will never be None after fitting
  lower_bound: Optional[float] = None
  upper_bound: Optional[float] = None

  def fit(self, data: pd.Series)->pd.Series:
    data = data.astype(float)
    if self.lower_bound is not None:
      data[data < self.lower_bound] = self.lower_bound
    if self.upper_bound is not None:
      data[data > self.upper_bound] = self.upper_bound
    return data

class CategoricalSchemaColumn(pydantic.BaseModel, BaseSchemaColumn):
  name: str
  type: Literal[SchemaColumnType.Categorical]
  min_frequency: int = 1

  def fit(self, data: pd.Series)->pd.Series:
    categorical_column: pd.Categorical = pd.Categorical(data)

    # Remove min_frequency
    category_frequencies: pd.Series = categorical_column.value_counts()
    filtered_category_frequencies: pd.Series = (category_frequencies[category_frequencies < self.min_frequency])
    for category in filtered_category_frequencies.index:
      categorical_column[categorical_column == category] = pd.NA
    return cast(pd.Series, categorical_column)

class UniqueSchemaColumn(pydantic.BaseModel, BaseSchemaColumn):
  name: str
  type: Literal[SchemaColumnType.Unique]
  def fit(self, data: pd.Series)->pd.Series:
    return data.astype(str)

class TextSchemaColumn(pydantic.BaseModel, BaseSchemaColumn):
  name: str
  type: Literal[SchemaColumnType.Text]
  preprocessing: PreprocessingConfig

  TOPIC_OUTLIER: ClassVar[str] = '-1'

  def fit(self, data: pd.Series)->pd.Series:
    isna_mask = data.isna()
    new_data = data.copy()
    new_data[isna_mask] = ''
    return new_data.astype(str)
  
  def get_topic_column(self):
    return f"{self.name}-topic"
  
  def get_preprocess_column(self):
    return f"{self.name}-preprocess"
  

SchemaColumn = Annotated[Union[UniqueSchemaColumn, CategoricalSchemaColumn, TextSchemaColumn, RangeSchemaColumn], pydantic.Field(discriminator="type")]

__all__ = [
  "BaseSchemaColumn",
  "SchemaColumn",
  "TextSchemaColumn",
  "UniqueSchemaColumn",
  "CategoricalSchemaColumn",
  "RangeSchemaColumn",
  "SchemaColumnType",
]