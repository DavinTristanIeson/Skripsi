from dataclasses import dataclass
import functools
from math import ceil
from typing import Any, Optional, Sequence, cast
import pandas as pd

from modules.logger import TimeLogger
from modules.project.cache import ProjectCacheManager
from modules.storage import CacheItem
from modules.config import Config

from .pagination import PaginationMeta, PaginationParams
from .filter import _TableFilterParams, TableSort
from .filter_variants import TableFilter

@dataclass
class TableEngine:
  config: Config

  @functools.cached_property
  def cache(self):
    return ProjectCacheManager().get(self.config.project_id)
  
  @classmethod
  def workspace_key(cls, filter: Optional[TableFilter], sort: Optional[TableSort])->str:
    import hashlib
    return ' '.join(map(
      lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest(),
      [filter, sort]
    ))
  
  def save_to_cache(self, df: pd.DataFrame, filter: Optional[TableFilter], sort: Optional[TableSort]):
    key = self.workspace_key(filter, sort)
    self.cache.workspaces.set(CacheItem(
      key=key,
      value=df
    ))

  def get_cached_workspace(self, filter: Optional[TableFilter], sort: Optional[TableSort]):
    key = self.workspace_key(filter, sort)
    return self.cache.workspaces.get(key)
    
  def filter(self, df: pd.DataFrame, filter: Optional[TableFilter])->pd.DataFrame:
    if filter is None:
      return df
    with TimeLogger("TableEngine", title="Applying filter to dataset..."):
      mask = filter.apply(_TableFilterParams(
        config=self.config,
        data=df
      ))
      return df[mask]
  
  def sort(self, df: pd.DataFrame, sort: Optional[TableSort])->pd.DataFrame:
    if sort is None:
      return df
    with TimeLogger("TableEngine", title="Applying sort to dataset..."):
      return df.sort_values(by=sort.name, ascending=sort.asc)
    

  def process_workspace(self, filter: Optional[TableFilter], sort: Optional[TableSort])->pd.DataFrame:
    df = self.cache.load_workspace()

    config_column_names = map(lambda x: x.name, self.config.data_schema.columns)
    column_names = [col for col in config_column_names if col in df.columns]
    df = df.loc[:, column_names]

    cached_df = self.get_cached_workspace(filter, sort)
    if cached_df is not None:
      return cached_df

    if filter is not None:
      cached_filtered_df = self.get_cached_workspace(filter, None)
      if cached_filtered_df is not None:
        df = cached_filtered_df
      else:
        df = self.filter(df, filter)
        self.save_to_cache(df, filter, None)
    
    if sort is not None:
      df = self.sort(df, sort)
      self.save_to_cache(df, filter, sort)

    return df
      
  def get_meta(self, df: Sequence[Any], params: PaginationParams)->PaginationMeta:
    if params.limit is None:
      return PaginationMeta(
        pages=1,
        total=len(df)
      )
    pages = ceil(len(df) / params.limit)
    return PaginationMeta(
      pages=pages,
      total=len(df)
    )
    
  def paginate_workspace(self, params: PaginationParams):
    df = self.process_workspace(params.filter, params.sort)

    meta = self.get_meta(cast(Sequence[Any], df), params)
    if params.limit is not None:
      page = params.page or 0
      from_idx = page * params.limit
      to_idx = (page + 1) * params.limit
      df = df.iloc[from_idx:to_idx, :]

    df["__index"] = df.index

    return df, meta

    
__all__ = [
  "TableEngine"
]