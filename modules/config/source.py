import abc
from enum import Enum
from typing import Annotated, Literal, Optional, Union

import pydantic
import pandas as pd

from modules.logger import ProvisionedLogger
from modules.validation import DiscriminatedUnionValidator
from modules.api import ExposedEnum

class DataSourceTypeEnum(str, Enum):
  CSV = "csv"
  Parquet = "parquet"
  Excel = "excel"

ExposedEnum().register(DataSourceTypeEnum)

class _BaseDataSource(pydantic.BaseModel, abc.ABC, frozen=True):
  path: str

  @abc.abstractmethod
  def load(self)->pd.DataFrame:
    pass

logger = ProvisionedLogger().provision("Config")

class CSVDataSource(_BaseDataSource, pydantic.BaseModel, frozen=True):
  type: Literal[DataSourceTypeEnum.CSV]
  delimiter: str = ','

  def load(self)->pd.DataFrame:
    df = pd.read_csv(self.path, delimiter=self.delimiter, on_bad_lines="skip", encoding='utf-8')
    logger.info(f"Loaded data source from {self.path}")
    return df

class ParquetDataSource(_BaseDataSource, pydantic.BaseModel, frozen=True):
  type: Literal[DataSourceTypeEnum.Parquet]
  def load(self)->pd.DataFrame:
    df = pd.read_parquet(self.path)
    logger.info(f"Loaded data source from {self.path}")
    return df
  
class ExcelDataSource(_BaseDataSource, pydantic.BaseModel, frozen=True):
  type: Literal[DataSourceTypeEnum.Excel]
  sheet_name: Optional[str]
  
  def load(self)->pd.DataFrame:
    kwargs = dict()
    if self.sheet_name is not None:
      kwargs["sheet_name"] = self.sheet_name
    df = pd.read_excel(self.path)
    logger.info(f"Loaded data source from {self.path}")
    return df

# Definitely should be frozen. They should be stable since they're going to be used with lru_cache.
DataSource = Annotated[Union[CSVDataSource, ParquetDataSource, ExcelDataSource], pydantic.Field(discriminator="type"), DiscriminatedUnionValidator]

__all__ = [
  "DataSourceTypeEnum",
  "ExcelDataSource",
  "CSVDataSource",
  "ParquetDataSource",
  "DataSource",
]