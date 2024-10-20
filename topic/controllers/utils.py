
import pandas as pd
from common.models.api import ApiError

def assert_column_exists(df: pd.DataFrame, col: str)->pd.Series:
  if col not in df.columns:
    raise ApiError(f"Cannot find column {col} in the workspace table. Perhaps the cached results are outdated. Please re-execute the topic modeling procedure.", 404)
  return df[col]