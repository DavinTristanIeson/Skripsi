from dataclasses import dataclass
import functools
from math import ceil
from typing import Any, Optional, Sequence
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
    return ' '.join(map(
      lambda x: hex(hash(x)).lstrip('-').lstrip('0x'),
      [filter, sort]
    ))
  
  def save_to_cache(self, df: pd.DataFrame, filter: Optional[TableFilter], sort: Optional[TableSort]):
    key = self.workspace_key(filter, sort)
    self.cache.workspaces.set(CacheItem(
      key=key,
      value=df
    ))
    
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
    
  def reorder(self, df: pd.DataFrame):
    return self.config.data_schema.process_columns(df)
  
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
    
  def paginate(self, df: pd.DataFrame, params: PaginationParams):
    column_names = map(lambda x: x.name, self.config.data_schema.columns)
    existent_column_names = filter(lambda col: col in df.columns, column_names)
    columns = list(existent_column_names)

    df = df.loc[:,columns]
    if params.filter is not None:
      filtered_df_key = self.workspace_key(params.filter, None)
      cached_df = self.cache.workspaces.get(filtered_df_key)
      if cached_df is None:
        df = self.filter(df, params.filter)
        self.save_to_cache(df, params.filter, None)
      else:
        df = cached_df
    if params.sort is not None:
      filtered_sorted_df_key = self.workspace_key(params.filter, params.sort)
      cached_df = self.cache.workspaces.get(filtered_sorted_df_key)
      if cached_df is None:
        df = self.sort(df, params.sort)
        self.save_to_cache(df, params.filter, params.sort)
      else:
        df = cached_df
    if params.limit is not None:
      page = params.page or 0
      from_idx = page * params.limit
      to_idx = (page + 1) * params.limit
      df = df.iloc[from_idx:to_idx, :]

    return self.reorder(df)
    
__all__ = [
  "TableEngine"
]