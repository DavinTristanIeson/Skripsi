from typing import Any, Literal, Union, cast, overload
import pandas as pd
import numpy as np
import numpy.typing as npt

# https://stackoverflow.com/questions/8904694/how-to-normalize-a-2-dimensional-numpy-array-in-python-less-verbose
@overload
def normalize_frequency(array: npt.NDArray, axis: Literal[0, 1, None])->npt.NDArray: ...
@overload
def normalize_frequency(array: pd.DataFrame, axis: Literal[0, 1, None])->pd.DataFrame: ...
def normalize_frequency(array: Union[npt.NDArray, pd.DataFrame], axis: Literal[0, 1, None]):
  # Row-wise normalization
  if axis == 0:
    row_sums = array.sum(axis=1)
    return array / row_sums.to_numpy()[:, np.newaxis]
  elif axis == 1:
    col_sums = array.sum(axis=0)
    return array / col_sums
  else:
    if isinstance(array, pd.DataFrame):
      grand_sum = array.sum(axis=1).sum()
    else:
      grand_sum = array.sum()
    return array / grand_sum
  

def pearson_residual_table(a: pd.Series, b: pd.Series)->pd.DataFrame:
  contingency_table = pd.crosstab(a, b)
  observed = normalize_frequency(contingency_table, None)

  marginal_total_x = observed.sum(axis=0).to_numpy()
  marginal_total_x = marginal_total_x.reshape((1, len(marginal_total_x)))
  
  marginal_total_y = observed.sum(axis=1).to_numpy()
  marginal_total_y = marginal_total_y.reshape((len(marginal_total_y), 1))

  expected = marginal_total_y @ marginal_total_x
  residuals = observed - expected
  indexed_residuals: npt.NDArray = residuals / np.sqrt(expected)

  return pd.DataFrame(indexed_residuals, index=observed.index, columns=observed.columns)

def binary_category(series: pd.Series, checked_value: Any, if_true: Any, if_false: Any):
  true_mask = series == checked_value
  copied = cast(pd.Series, pd.Categorical(series, categories=[if_true, if_false]))
  copied[true_mask] = if_true
  copied[~true_mask] = if_false
  return copied