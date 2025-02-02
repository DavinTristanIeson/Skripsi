import abc
from enum import Enum
import functools
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

class BaseDataSource(abc.ABC):
  path: FilePathField

  @abc.abstractmethod
  def load(self)->pd.DataFrame:
    pass

logger = RegisteredLogger().provision("Config")

@functools.lru_cache(1)
def load_csv(path: str, delimiter: str):
  df = pd.read_csv(path, delimiter=delimiter, on_bad_lines="skip")
  return df

class CSVDataSource(pydantic.BaseModel, BaseDataSource):
  model_config = CommonModelConfig
  type: Literal[DataSourceTypeEnum.CSV]
  delimiter: str = ','

  def load(self)->pd.DataFrame:
    df = load_csv(self.path, self.delimiter)
    logger.info(f"Loaded data source from {self.path}")
    return df

@functools.lru_cache(1)
def load_parquet(path: str)->pd.DataFrame:
  df = pd.read_parquet(path)
  return df

class ParquetDataSource(pydantic.BaseModel, BaseDataSource):
  model_config = CommonModelConfig
  type: Literal[DataSourceTypeEnum.Parquet]
  def load(self)->pd.DataFrame:
    df = load_parquet(self.path)
    logger.info(f"Loaded data source from {self.path}")
    return df
  
@functools.lru_cache(1)
def load_excel(path: str, *, sheet_name: str)->pd.DataFrame:
  kwargs = dict()
  if sheet_name is not None:
    kwargs["sheet_name"] = sheet_name
  df = pd.read_excel(path)
  return df
class ExcelDataSource(pydantic.BaseModel, BaseDataSource):
  model_config = CommonModelConfig
  type: Literal[DataSourceTypeEnum.Excel]
  sheet_name: Optional[str]
  
  def load(self)->pd.DataFrame:
    df = load_excel(self.path, sheet_name=self.sheet_name)
    logger.info(f"Loaded data source from {self.path}")
    return df

DataSource = Annotated[Union[CSVDataSource, ParquetDataSource, ExcelDataSource], pydantic.Field(discriminator="type"), DiscriminatedUnionValidator]

__all__ = [
  "DataSourceTypeEnum",
  "ExcelDataSource",
  "CSVDataSource",
  "DataSource"
]