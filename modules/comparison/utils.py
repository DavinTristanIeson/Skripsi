import pandas as pd

def _chisq_prepare(A: pd.Series, B: pd.Series, *, with_correction: bool = False):
  A_freq = A.value_counts()
  A_freq.name = A.name
  B_freq = B.value_counts()
  B_freq.name = B.name
  crosstab = pd.concat([A_freq, B_freq], axis=1)
  crosstab.fillna(0, inplace=True)

  if not with_correction:
    return crosstab

  empty_cells_count = (crosstab == 0).sum().sum()
  has_empty_cells = empty_cells_count > 0
  if has_empty_cells:
    # Haldene-Anscombe correction
    crosstab += 0.5
  return crosstab

def _mann_whitney_u_prepare(A: pd.Series, B: pd.Series):
  if A.dtype == 'category':
    A = A.cat.codes
  if B.dtype == 'category':
    B = B.cat.codes
  return A.to_numpy(), B.to_numpy()