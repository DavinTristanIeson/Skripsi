import numpy as np
import pandas as pd
import scipy.stats

from modules.comparison.exceptions import NotMutuallyExclusiveException

def _chisq_prepare_contingency_table(contingency_table: pd.DataFrame, *, with_correction: bool = False):
  contingency_table = contingency_table.fillna(0)
  if not with_correction:
    return contingency_table

  empty_cells_count = (contingency_table == 0).sum().sum()
  has_empty_cells = empty_cells_count > 0
  if has_empty_cells:
    # Haldene-Anscombe correction
    contingency_table += 0.5
  return contingency_table

def _chisq_prepare(groups: list[pd.Series], *, with_correction: bool = False):
  group_frequencies: list[pd.Series] = []
  for group in groups:
    frequencies = group.value_counts()
    frequencies.name = group.name
    group_frequencies.append(frequencies)

  crosstab = pd.concat(group_frequencies, axis=1)
  crosstab = crosstab.fillna(0)
  return _chisq_prepare_contingency_table(crosstab, with_correction=with_correction)

def _mann_whitney_u_prepare(groups: list[pd.Series])->list[np.ndarray]:
  return list(map(
    lambda data: data.to_numpy(),
    map(
      lambda data: (data.cat.codes if data.dtype == 'category' else data),
      groups
    )
  ))

def _check_chisq_contingency_table(contingency_table: pd.DataFrame):
  less_than_5 = contingency_table < 5
  less_than_5_indices = np.argwhere(less_than_5)
  less_than_5_count: int = less_than_5.sum().sum()

  warnings: list[str] = []
  if less_than_5_count >= less_than_5.size * 0.2:
    observations: list[str] = []
    for index in less_than_5_indices:
      group_name = contingency_table.columns[index[1]]
      row_name = contingency_table.index[index[0]]
      observations.append(f"{group_name} - {row_name} ({contingency_table.iat[index[0], index[1]]} obs.)")

    warnings.append(f"More than 20% of the contingency table has less than 5 observations: {', '.join(observations[:5])}")
  return warnings


def _check_normal_distribution(groups: list[pd.Series], name: str)->list[str]:
  warnings = []
  for data in groups:
    if len(data) < 30:
      warnings.append(f"There are too few samples in {data.name} ({len(data)} rows) to expect that the data follows a normal distribution.")
    else:
      normaltest_result = scipy.stats.normaltest(data).pvalue
      is_normal = normaltest_result < 0.05
      if not is_normal:
        warnings.append(f"{name} expects the samples to be normally distributed, but \"{data.name}\" does not follow a normal distribution (confidence: {(1 - normaltest_result)*100:.2f}%).")
  return warnings


def _check_non_normal_distribution(groups: list[pd.Series], name: str)->list[str]:
  warnings = []
  for data in groups:
    if not pd.api.types.is_numeric_dtype(data.dtype):
      continue
    
    if len(data) >= 30:
      normaltest_result = scipy.stats.normaltest(data).pvalue
      is_normal = normaltest_result < 0.05
      if is_normal:
        warnings.append(f"{name} is generally used when the samples do not follow a normal distribution, but \"{data.name}\" does follow a normal distribution (p-value: {is_normal}). Consider using parametric statistic tests instead.")

  return warnings

def cramer_v(contingency_table: pd.DataFrame):
  # Cramer V with bias correction (https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V)
  chi2_result = scipy.stats.chi2_contingency(contingency_table)
  chi2 = chi2_result.statistic # type: ignore

  n = contingency_table.sum().sum()
  psi2 = chi2 / n
  k = contingency_table.shape[1]
  r = contingency_table.shape[0]

  k_tilde = k - (np.power(k - 1, 2) / (n - 1))
  r_tilde = r - (np.power(r - 1, 2) / (n - 1))
  psi2_tilde = max(0, psi2 - ((k-1) * (r-1) / (n-1)))

  V = np.sqrt(psi2_tilde / min(k_tilde - 1, r_tilde - 1))
  return V

def assert_mutually_exclusive(groups: list[pd.Series]):
  if len(groups) == 0:
    return
  for i in range(len(groups)):
    for j in range(i + 1, len(groups)):
      data_A = groups[i].index
      data_B = groups[j].index
      overlap = data_A.intersection(data_B) # type: ignore
      if not overlap.empty:
        raise NotMutuallyExclusiveException(
          group1=str(groups[i].name),
          group2=str(groups[j].name),
          overlap_count=len(overlap)
        )
