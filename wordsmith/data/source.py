import abc
from enum import Enum
import importlib
import importlib.machinery
from types import ModuleType
from typing import Annotated, Literal, Optional, Union, cast
import os
import importlib.util

import pydantic
import pandas as pd

from common.logger import RegisteredLogger
from common.utils.loader import hashfile
from common.models.api import ApiError

class DataSourceTypeEnum(str, Enum):
  CSV = "csv"
  Parquet = "parquet"
  Excel = "excel"
  Python = "python"

class BaseDataSource(abc.ABC):
  path: str

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

# Not recommended    
class PythonDataSource(pydantic.BaseModel, BaseDataSource):
  type: Literal[DataSourceTypeEnum.Python]
  variable_name: str
  def load(self)->pd.DataFrame:
    file_path = os.path.normpath(self.path)
    module_path = file_path.replace(os.path.sep, '.')
    # module = importlib.import_module(module_path)

    # https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
    # For 3.5 and above
    spec: Optional[importlib.machinery.ModuleSpec] = importlib.util.spec_from_file_location(module_path, file_path)
    if spec is None or spec.loader is None:
      raise ApiError(f"Failed to find a valid python module in {file_path}", 404)
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if self.variable_name is None:
      raise ApiError("variable_name field is not provided for data source type Python", 422)
    try:
      df = cast(pd.DataFrame, module.__dict__[self.variable_name])
      logger.info(f"Loaded data source from {self.path}")
      return df
    except ModuleNotFoundError:
      raise ApiError(f"Failed to load {self.path} as a Python module", 404)
    except KeyError:
      raise ApiError(f"The python script at {self.path} does not seem to contain a global variable with name {self.variable_name}", 422)
  

DataSource = Annotated[Union[PythonDataSource, CSVDataSource, ParquetDataSource, ExcelDataSource], pydantic.Field(discriminator="type")]


__all__ = [
  "DataSourceTypeEnum",
  "PythonDataSource",
  "ExcelDataSource",
  "CSVDataSource",
  "DataSource"
]