from dataclasses import dataclass
import functools
from math import ceil
from typing import Any, Optional, Sequence, cast
import pandas as pd

from modules.logger import TimeLogger
from modules.project.cache_manager import ProjectCacheManager
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
    if sort is None or sort.name not in df.columns:
      return df
    with TimeLogger("TableEngine", title="Applying sort to dataset..."):
      if pd.api.types.is_bool_dtype(df[sort.name]):
        # Invert ascending order. True boolean values should be ranked before False boolean values.
        ascending = not sort.asc
      else:
        ascending = sort.asc
      return df.sort_values(by=sort.name, ascending=ascending)
    

  def process_workspace(self, filter: Optional[TableFilter], sort: Optional[TableSort])->pd.DataFrame:
    df = self.cache.workspaces.load()

    config_column_names = map(lambda x: x.name, self.config.data_schema.columns)
    column_names = [col for col in config_column_names if col in df.columns]
    df = df.loc[:, column_names]

    cached_df = self.cache.workspaces.get(
      self.workspace_key(filter, sort)
    )
    if cached_df is not None:
      return cached_df

    if filter is not None:
      cached_filtered_df = self.cache.workspaces.get(
        self.workspace_key(filter, None)
      )
      if cached_filtered_df is not None:
        df = cached_filtered_df
      else:
        df = self.filter(df, filter)
        self.cache.workspaces.set(df, self.workspace_key(filter, None))
    
    if sort is not None:
      df = self.sort(df, sort)
      self.cache.workspaces.set(df, self.workspace_key(filter, sort))

    df = self.config.data_schema.process_columns(df)
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