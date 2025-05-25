from typing import Optional, Sequence, cast
import pandas as pd

from modules.regression.exceptions import ReferenceMustBeAValidSubdatasetException

def is_boolean_dataframe_mutually_exclusive(df: pd.DataFrame):
  agg_column = df.sum(axis=1)
  return (agg_column != 1).sum() == 0

def one_hot_to_effect_coding(df: pd.DataFrame, reference: Optional[str]=None):
  levels = df.columns
  if reference is None:
    reference = sorted(levels)[-1]  # Default to last alphabetically
  elif reference not in levels:
    raise ReferenceMustBeAValidSubdatasetException(
      reference=reference,
      groups=cast(Sequence[str], levels),
    )
  
  # Columns to keep for effect coding (exclude reference)
  effect_cols = list(filter(lambda lvl: lvl != reference, levels))
  effect_df = df.loc[:, effect_cols]

  ref_mask = df[reference] == 1
  effect_df.loc[ref_mask, :] = -1
  return effect_df
