from dataclasses import dataclass
import functools
from typing import Sequence
import numpy as np
import pandas as pd
from common.logger import TimeLogger
from common.storage.cache import CacheItem
from models.config import Config
from models.project.cache import ProjectCacheManager
from models.table.errors import TableFilterError
from models.table.pagination import PaginationParams
from .filter import TableFilterParams, TableSort
from .filter_variants import TableFilter

@dataclass
class TableEngine:
  config: Config
  @functools.cached_property
  def cache(self):
    return ProjectCacheManager().get(self.config.project_id)
  def load_workspace(self):
    cache = self.cache
    empty_key = cache.workspace_key([], [])
    cached_df = cache.workspaces.get(empty_key)
    if cached_df is not None:
      return cached_df
    
    df = self.config.paths.load_workspace()
    cache.workspaces.set(CacheItem(
      key=empty_key,
      value=df,
      persistent=True
    ))
    return df

  def filter(self, filters: Sequence[TableFilter], sorts: Sequence[TableSort])->pd.DataFrame:
    df = self.load_workspace()
    mask = np.full(len(df), 1, dtype=np.bool_)
    with TimeLogger("DatasetFilter", title="Applying filter to dataset...", report_start=True):
      for filter in filters:
        if filter.target not in df.columns:
          raise TableFilterError.ColumnNotFound(
            project_id=self.config.project_id,
            target=filter.target
          )

        mask &= filter.apply(TableFilterParams(
          column=self.config.data_schema.assert_exists(filter.target),
          data=df[filter.target],
        ))

    cache = self.cache
    cache.workspaces.set(CacheItem(
      key=cache.workspace_key(filters, sorts),
      value=df,
      persistent=True
    ))
    return df[mask]

  def paginate(self, df: pd.DataFrame, params: PaginationParams):
    df = self.filter(params.filter, params.sort)
    columns = list(map(lambda x: x.name, filter(lambda x: x.active, self.config.data_schema.columns)))
    from_idx = params.page * params.limit
    to_idx = (params.page + 1) * params.limit
    return df.loc[from_idx:to_idx, columns]