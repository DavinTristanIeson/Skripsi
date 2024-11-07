import abc
import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Sequence, Union, cast
import numpy as np
import pydantic
import pandas as pd

from common.models.validators import DiscriminatedUnionValidator, CommonModelConfig, FilenameField
from common.models.enum import ExposedEnum
from wordsmith.data.textual import TextPreprocessingConfig, TopicModelingConfig

class SchemaColumnTypeEnum(str, Enum):
  Continuous = "continuous"
  Categorical = "categorical"
  Temporal = "temporal"
  Textual = "textual"
  Unique = "unique"

ExposedEnum().register(SchemaColumnTypeEnum)

class FillNaModeEnum(str, Enum):
  ForwardFill = "ffill",
  BackwardFill = "bfill",
  Value = "value"
  Exclude = "exclude"

  @staticmethod
  def fillna(df: pd.Series, mode: Optional["FillNaModeEnum"], value: Optional[Any]):
    if mode == FillNaModeEnum.ForwardFill:
      return df.ffill()
    elif mode == FillNaModeEnum.BackwardFill:
      return df.bfill()
    elif value is not None:
      return df.fillna(value)
    else:
      return df

ExposedEnum().register(FillNaModeEnum)

class BaseSchemaColumn(pydantic.BaseModel, abc.ABC):
  name: FilenameField
  dataset_name: Optional[str] = None

  @abc.abstractmethod
  def fit(self, data: pd.Series)->pd.Series:
    pass
class ContinuousSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Continuous]
  # Will never be None after fitting
  lower_bound: Optional[float] = None
  upper_bound: Optional[float] = None

  fill_na: Optional[FillNaModeEnum] = FillNaModeEnum.Exclude
  fill_na_value: Optional[float] = None

  def fit(self, data: pd.Series)->pd.Series:
    data = data.astype(np.float64)
    if self.lower_bound is not None:
      data[data < self.lower_bound] = self.lower_bound
    if self.upper_bound is not None:
      data[data > self.upper_bound] = self.upper_bound

    data = FillNaModeEnum.fillna(data, self.fill_na, self.fill_na_value)
    return data

class CategoricalSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Categorical]
  min_frequency: int = pydantic.Field(default=1, gt=0)

  fill_na: Optional[FillNaModeEnum] = FillNaModeEnum.Exclude
  fill_na_value: Optional[str] = None

  def fit(self, data: pd.Series)->pd.Series:
    # Remove min_frequency
    data = data.copy()
    category_frequencies: pd.Series = data.value_counts()
    filtered_category_frequencies: pd.Series = (category_frequencies[category_frequencies < self.min_frequency])
    for category in filtered_category_frequencies.index:
      data[data == category] = pd.NA

    data = FillNaModeEnum.fillna(data, FillNaModeEnum.Value, self.fill_na_value)
    return cast(pd.Series, pd.Categorical(data))

class UniqueSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  type: Literal[SchemaColumnTypeEnum.Unique]
  def fit(self, data: pd.Series)->pd.Series:
    return data

class TextualSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Textual]
  preprocessing: TextPreprocessingConfig
  topic_modeling: TopicModelingConfig

  def fit(self, data: pd.Series)->pd.Series:
    isna_mask = data.isna()
    new_data = data.astype(str)
    new_data[isna_mask] = ''
    documents = cast(Sequence[str], new_data[~isna_mask])

    preprocessed_documents = tuple(map(lambda x: ' '.join(x), self.preprocessing.preprocess(documents)))
    new_data[~isna_mask] = preprocessed_documents
    return new_data
  
  @property
  def topic_column(self):
    return f"__wordsmith_topic_{self.name}"
  
  @property
  def topic_index_column(self):
    return f"__wordsmith_topic_index_{self.name}"
  
  @property
  def preprocess_column(self):
    return f"__wordsmith_preprocess_{self.name}"
  
class TemporalSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Temporal]
  min_date: Optional[datetime.datetime]
  max_date: Optional[datetime.datetime]
  bins: int = 15
  datetime_format: Optional[str]

  fill_na: Optional[FillNaModeEnum] = None
  fill_na_value: Optional[datetime.datetime] = None

  def fit(self, data: pd.Series)->pd.Series:
    kwargs = dict()
    if self.datetime_format is not None:
      kwargs["format"] = self.datetime_format
    datetime_column = pd.to_datetime(data, **kwargs)
    if self.min_date is not None:
      datetime_column[datetime_column < self.min_date] = self.min_date
    if self.max_date is not None:
      datetime_column[datetime_column > self.max_date] = self.max_date

    data = FillNaModeEnum.fillna(data, self.fill_na, self.fill_na_value)
    return datetime_column

SchemaColumn = Annotated[Union[UniqueSchemaColumn, CategoricalSchemaColumn, TextualSchemaColumn, ContinuousSchemaColumn, TemporalSchemaColumn], pydantic.Field(discriminator="type"), DiscriminatedUnionValidator]

__all__ = [
  "BaseSchemaColumn",
  "SchemaColumn",
  "TextualSchemaColumn",
  "UniqueSchemaColumn",
  "CategoricalSchemaColumn",
  "ContinuousSchemaColumn",
  "SchemaColumnTypeEnum",
]