import abc
import datetime
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, Optional, Sequence, Union, cast
import pydantic
import pandas as pd

from common.models.validators import DiscriminatedUnionValidator, CommonModelConfig, FilenameField
from common.models.enum import EnumMemberDescriptor, ExposedEnum
from wordsmith.data.textual import TextPreprocessingConfig, TopicModelingConfig

class SchemaColumnTypeEnum(str, Enum):
  Continuous = "continuous"
  Categorical = "categorical"
  Temporal = "temporal"
  Textual = "textual"
  Unique = "unique"

ExposedEnum().register(SchemaColumnTypeEnum, {
  SchemaColumnTypeEnum.Continuous: EnumMemberDescriptor(
    label="Continuous",
  ),
  SchemaColumnTypeEnum.Categorical: EnumMemberDescriptor(
    label="Categorical"
  ),
  SchemaColumnTypeEnum.Temporal: EnumMemberDescriptor(
    label="Temporal",
  ),
  SchemaColumnTypeEnum.Textual: EnumMemberDescriptor(
    label="Textual",
  ),
  SchemaColumnTypeEnum.Unique: EnumMemberDescriptor(
    label="Unique",
  ),
})

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

  def fit(self, data: pd.Series)->pd.Series:
    data = data.astype(float)
    if self.lower_bound is not None:
      data[data < self.lower_bound] = self.lower_bound
    if self.upper_bound is not None:
      data[data > self.upper_bound] = self.upper_bound
    return data

class CategoricalSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Categorical]
  min_frequency: int = 1

  def fit(self, data: pd.Series)->pd.Series:
    categorical_column: pd.Categorical = pd.Categorical(data)

    # Remove min_frequency
    category_frequencies: pd.Series = categorical_column.value_counts()
    filtered_category_frequencies: pd.Series = (category_frequencies[category_frequencies < self.min_frequency])
    for category in filtered_category_frequencies.index:
      categorical_column[categorical_column == category] = pd.NA
    return cast(pd.Series, categorical_column)

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
    return f"__topic_{self.name}"
  
  @property
  def preprocess_column(self):
    return f"__preprocess_{self.name}"
  
class TemporalSchemaColumn(BaseSchemaColumn, pydantic.BaseModel):
  model_config = CommonModelConfig
  type: Literal[SchemaColumnTypeEnum.Temporal]
  min_date: Optional[datetime.datetime]
  max_date: Optional[datetime.datetime]
  bins: int = 15
  datetime_format: Optional[str]

  def fit(self, data: pd.Series)->pd.Series:
    kwargs = dict()
    if self.datetime_format is not None:
      kwargs["format"] = self.datetime_format
    datetime_column = pd.to_datetime(data, **kwargs)
    if self.min_date is not None:
      datetime_column[datetime_column < self.min_date] = self.min_date
    if self.max_date is not None:
      datetime_column[datetime_column > self.max_date] = self.max_date
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