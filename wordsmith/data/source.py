import abc
from enum import Enum
import importlib.machinery
from typing import Annotated, Literal, Optional, Union
import importlib.util

import pydantic
import pandas as pd

from common.logger import RegisteredLogger
from common.models.enum import EnumMemberDescriptor, ExposedEnum
from common.utils.loader import hashfile

class DataSourceTypeEnum(str, Enum):
  CSV = "csv"
  Parquet = "parquet"
  Excel = "excel"

ExposedEnum().register(DataSourceTypeEnum, {
  DataSourceTypeEnum.CSV: EnumMemberDescriptor(
    label="CSV",
  ),
  DataSourceTypeEnum.Parquet: EnumMemberDescriptor(
    label="Parquet",
  ),
  DataSourceTypeEnum.Excel: EnumMemberDescriptor(
    label="XLSX / Excel",
  ),
})

FilePath = Annotated[str, pydantic.Field(pattern=r"^[a-zA-Z0-9-_. \/\\]+$")]
class BaseDataSource(abc.ABC):
  path: FilePath

  @abc.abstractmethod
  def load(self)->pd.DataFrame:
    pass

  def hash(self)->str:
    return hashfile(self.path)

logger = RegisteredLogger().provision("Config")
class CSVDataSource(pydantic.BaseModel, BaseDataSource):
  type: Literal[DataSourceTypeEnum.CSV]
  delimiter: str = ','
  limit: Optional[int] = None

  def skiprows(self, i: int):
    if self.limit is None:
      return False
    return i > self.limit

  def load(self)->pd.DataFrame:
    kwargs = dict()
    if self.limit is not None:
      kwargs["skiprows"] = self.skiprows
    df = pd.read_csv(self.path, delimiter=self.delimiter, on_bad_lines="skip", **kwargs)
    logger.info(f"Loaded data source from {self.path}")
    return df
  
class ParquetDataSource(pydantic.BaseModel, BaseDataSource):
  type: Literal[DataSourceTypeEnum.Parquet]
  def load(self)->pd.DataFrame:
    return pd.read_parquet(self.path)
  
class ExcelDataSource(pydantic.BaseModel, BaseDataSource):
  type: Literal[DataSourceTypeEnum.Excel]
  sheet_name: Optional[str]
  def load(self)->pd.DataFrame:
    kwargs = dict()
    if self.sheet_name is not None:
      kwargs["sheet_name"] = self.sheet_name
    df = pd.read_excel(self.path, **kwargs)
    logger.info(f"Loaded data source from {self.path}")
    return df

DataSource = Annotated[Union[CSVDataSource, ParquetDataSource, ExcelDataSource], pydantic.Field(discriminator="type")]

__all__ = [
  "DataSourceTypeEnum",
  "ExcelDataSource",
  "CSVDataSource",
  "DataSource"
]