import abc
from enum import Enum
from typing import Annotated, Literal, Optional, Union

import pydantic
import pandas as pd

from common.logger import RegisteredLogger
from common.models.validators import CommonModelConfig, discriminated_union_model_validator
from common.models.enum import ExposedEnum

class DataSourceTypeEnum(str, Enum):
  CSV = "csv"
  Parquet = "parquet"
  Excel = "excel"

ExposedEnum().register(DataSourceTypeEnum)

FilePath = Annotated[str, pydantic.Field(pattern=r"^[a-zA-Z0-9-_. \/\\:]+$")]
class BaseDataSource(abc.ABC):
  path: FilePath
  limit: Optional[int] = None

  @abc.abstractmethod
  def load(self)->pd.DataFrame:
    pass

logger = RegisteredLogger().provision("Config")
class CSVDataSource(pydantic.BaseModel, BaseDataSource):
  model_config = CommonModelConfig
  type: Literal[DataSourceTypeEnum.CSV]
  delimiter: str = ','

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
  model_config = CommonModelConfig
  type: Literal[DataSourceTypeEnum.Parquet]
  def load(self)->pd.DataFrame:
    return pd.read_parquet(self.path)[:self.limit]
  
class ExcelDataSource(pydantic.BaseModel, BaseDataSource):
  model_config = CommonModelConfig
  type: Literal[DataSourceTypeEnum.Excel]
  sheet_name: Optional[str]
  
  def skiprows(self, i: int):
    if self.limit is None:
      return False
    return i > self.limit
  
  def load(self)->pd.DataFrame:
    kwargs = dict()
    if self.sheet_name is not None:
      kwargs["sheet_name"] = self.sheet_name
    df = pd.read_excel(self.path, skiprows=self.skiprows, **kwargs) # type: ignore
    logger.info(f"Loaded data source from {self.path}")
    return df

DataSourceUnion = Annotated[Union[CSVDataSource, ParquetDataSource, ExcelDataSource], pydantic.Field(discriminator="type")]
class DataSource(pydantic.RootModel[DataSourceUnion]):
  _error_rewriter = discriminated_union_model_validator("type")

__all__ = [
  "DataSourceTypeEnum",
  "ExcelDataSource",
  "CSVDataSource",
  "DataSource"
]