from typing import cast
import scipy.stats
import pandas as pd
import math
import numpy.typing as npt
import numpy as np

def cramer_v(crosstab: pd.DataFrame)->tuple[float, float]:
  # References:
  # https://github.com/shakedzy/dython/blob/master/dython/nominal.py
  # https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V#:~:text=Bias%20correction%20Cram%C3%A9r%27s%20V%20can%20be%20a%20heavily,but%20with%20typically%20much%20smaller%20mean%20squared%20error.
  chisq, pvalue, _, _ = cast(tuple[float, float, float, npt.NDArray], scipy.stats.chi2_contingency(crosstab))
  rows = len(crosstab.index)
  cols = len(crosstab.columns)
  n = crosstab.sum().sum()

  corrected_chisq = max(0, (chisq / n) - ((cols-1) * (rows-1) / (n-1)))
  corrected_cols = cols - (pow(cols - 1, 2) / (n - 1))
  corrected_rows = rows - (pow(rows - 1, 2) / (n - 1))

  v_score = math.sqrt(
    float(corrected_chisq) /
    min(corrected_cols - 1, corrected_rows - 1)
  )
  return pvalue, v_score

def uncertainty(Pxy: pd.DataFrame):
  EPSILON = 1e-20
  Px = np.maximum(Pxy.sum(axis=1), EPSILON)
  Py = np.maximum(Pxy.sum(axis=0), EPSILON)
  Pxify = np.maximum(Pxy / Py, EPSILON)

  Hx = np.maximum(scipy.stats.entropy(Px), EPSILON)
  Hxy = -1 * (Pxy * np.log(Pxify)).sum().sum()
  Uxify = (Hx - Hxy) / Hx

  return Uxify