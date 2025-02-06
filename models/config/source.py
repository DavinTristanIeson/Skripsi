import abc
from enum import Enum
from typing import Annotated, Literal, Optional, Union

import pydantic
import pandas as pd

from common.logger import RegisteredLogger
from common.models.validators import CommonModelConfig, DiscriminatedUnionValidator, FilePathField
from common.models.enum import ExposedEnum

class DataSourceTypeEnum(str, Enum):
  CSV = "csv"
  Parquet = "parquet"
  Excel = "excel"

ExposedEnum().register(DataSourceTypeEnum)

class BaseDataSource(pydantic.BaseModel, abc.ABC, frozen=True):
  path: FilePathField

  @abc.abstractmethod
  def load(self)->pd.DataFrame:
    pass

logger = RegisteredLogger().provision("Config")

class CSVDataSource(BaseDataSource, pydantic.BaseModel, frozen=True):
  model_config = CommonModelConfig
  type: Literal[DataSourceTypeEnum.CSV]
  delimiter: str = ','

  def load(self)->pd.DataFrame:
    df = pd.read_csv(self.path, delimiter=self.delimiter, on_bad_lines="skip")
    logger.info(f"Loaded data source from {self.path}")
    return df

class ParquetDataSource(BaseDataSource, pydantic.BaseModel, frozen=True):
  model_config = CommonModelConfig
  type: Literal[DataSourceTypeEnum.Parquet]
  def load(self)->pd.DataFrame:
    df = pd.read_parquet(self.path)
    logger.info(f"Loaded data source from {self.path}")
    return df
  
class ExcelDataSource(BaseDataSource, pydantic.BaseModel, frozen=True):
  model_config = CommonModelConfig
  type: Literal[DataSourceTypeEnum.Excel]
  sheet_name: Optional[str]
  
  def load(self)->pd.DataFrame:
    kwargs = dict()
    if self.sheet_name is not None:
      kwargs["sheet_name"] = self.sheet_name
    df = pd.read_excel(self.path)
    logger.info(f"Loaded data source from {self.path}")
    return df

DataSource = Annotated[Union[CSVDataSource, ParquetDataSource, ExcelDataSource], pydantic.Field(discriminator="type"), DiscriminatedUnionValidator]

__all__ = [
  "DataSourceTypeEnum",
  "ExcelDataSource",
  "CSVDataSource",
  "DataSource",
]