import abc
from enum import Enum
import importlib
import importlib.machinery
from types import ModuleType
from typing import Annotated, Any, Callable, Literal, Optional, Sequence, Union, cast
import json
import os
import importlib.util

import pydantic
import pandas as pd

from wordsmith.data.common import DataSourceLoadingException
from wordsmith.utils.loader import hashfile

class DataSourceTypeEnum(str, Enum):
  Python = "python",
  Json = "json",
  CSV = "csv",

class BaseDataSource(abc.ABC):
  path: str
  limit: Optional[int] = None

  @abc.abstractmethod
  def load(self)->pd.DataFrame:
    pass

  def hash(self)->str:
    return hashfile(self.path)

class CSVDataSource(pydantic.BaseModel, BaseDataSource):
  type: Literal[DataSourceTypeEnum.CSV]
  delimiter: str = ','
  def load(self)->pd.DataFrame:
    skiprows: Union[Callable[[int], bool], Sequence[int]] = (lambda i: i > cast(int, self.limit)) if self.limit is not None else []
    return pd.read_csv(self.path, delimiter=self.delimiter, skiprows=skiprows, on_bad_lines="skip")
  
class JSONDataSource(pydantic.BaseModel, BaseDataSource):
  type: Literal[DataSourceTypeEnum.Json]
  json_field: Optional[str]
  def load(self)->pd.DataFrame:
    with open(self.path, encoding='utf-8') as f:
      contents = json.load(f)
      if self.json_field is None:
        try: iter(contents)
        except: raise DataSourceLoadingException(f"Failed to load {self.path} as JSON, since json_field is not specified, and the JSON file does not have an array at the top-level.")
        return pd.DataFrame(contents)
      
      paths = self.json_field.split('.')
      for path in paths:
        try:
          array = cast(Sequence[Any], contents[path])
        except KeyError:
          raise DataSourceLoadingException(f"Failed to follow {' -> '.join(paths)} in the JSON file. Please make sure that `json_field` contains a dot path to the actual data.")
        if self.limit is not None:
          try:
            array = array[:self.limit]
          except TypeError:
            raise DataSourceLoadingException(f"The value contained in {' -> '.join(paths)} is not an array.")
        contents = cast(Any, array)
      
      return pd.DataFrame(contents)
    
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
      raise DataSourceLoadingException(f"Failed to find a valid python module in {file_path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if self.variable_name is None:
      raise DataSourceLoadingException("variable_name field is not provided for data source type Python")
    try:
      return cast(pd.DataFrame, module.__dict__[self.variable_name])
    except ModuleNotFoundError:
      raise DataSourceLoadingException(f"Failed to load {self.path} as a Python module")
    except KeyError:
      raise DataSourceLoadingException(f"The python script at {self.path} does not seem to contain a global variable with name {self.variable_name}")
  

DataSource = Annotated[Union[PythonDataSource, CSVDataSource, JSONDataSource], pydantic.Field(discriminator="type")]


__all__ = [
  "DataSourceTypeEnum",
  "PythonDataSource",
  "JSONDataSource",
  "CSVDataSource",
  "DataSource"
]