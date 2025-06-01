import pandas as pd

from modules.comparison.base import EffectSizeResult, SignificanceResult
from modules.comparison.engine import TableComparisonEngine, StatisticTestResult
from modules.comparison.statistic_test import StatisticTestMethodEnum
from modules.config.schema.base import ANALYZABLE_SCHEMA_COLUMN_TYPES, CATEGORICAL_SCHEMA_COLUMN_TYPES
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache
from modules.table.engine import TableEngine
from modules.table.filter_variants import NamedTableFilter, NotTableFilter
from ..model import (
  BinaryStatisticTestOnContingencyTableResultMainResource,
  BinaryStatisticTestOnContingencyTableResultResource,
  BinaryStatisticTestOnContingencyTableSchema,
  BinaryStatisticTestOnDistributionResultResource,
  BinaryStatisticTestOnDistributionSchema,
)
from routes.table.controller.preprocess import TablePreprocessModule

def binary_statistic_test_on_distribution(cache: ProjectCache, input: BinaryStatisticTestOnDistributionSchema):
  preprocess = TablePreprocessModule(cache)
  column = preprocess.assert_column(input.column, supported_types=ANALYZABLE_SCHEMA_COLUMN_TYPES)
  workspace = cache.workspaces.load()

  results: list[StatisticTestResult] = []
  with ProvisionedLogger().disable(["TableEngine", "TableComparisonEngine"]):
    for group in input.groups:
      anti_group = NamedTableFilter(
        name=f"NOT {group.name}",
        filter=NotTableFilter(operand=group.filter)
      )
      engine = TableComparisonEngine(
        config=cache.config,
        groups=[group, anti_group],
        exclude_overlapping_rows=False,
      )
      result = engine.compare(
        workspace,
        column_name=input.column,
        statistic_test_preference=input.statistic_test_preference,
        effect_size_preference=input.effect_size_preference,
      )
      results.append(result)
  return BinaryStatisticTestOnDistributionResultResource(
    column=column,
    groups=list(map(lambda group: group.name, input.groups)),
    results=results,
  )

def binary_statistic_test_on_contingency_table(cache: ProjectCache, input: BinaryStatisticTestOnContingencyTableSchema):
  import scipy.stats

  preprocess = TablePreprocessModule(
    cache=cache
  )
  column = preprocess.assert_column(input.column, CATEGORICAL_SCHEMA_COLUMN_TYPES)
  engine = TableEngine(config=cache.config)
  df = preprocess.load_dataframe(filter=None)
  data, df = preprocess.get_data(df, column=column, exclude_invalid=True, transform_data=True)
  subdataset_names = list(map(lambda group: group.name, input.groups))
  subdataset_masks = list(map(lambda group: engine.filter_mask(df, group.filter), input.groups))
  category_names = list(map(str, data.unique()))
  category_masks = list(map(lambda unique: data == unique, df[column.name].unique()))

  results: list[list[BinaryStatisticTestOnContingencyTableResultResource]] = []
  for variable1, mask1 in zip(subdataset_names, subdataset_masks):
    results_row: list[BinaryStatisticTestOnContingencyTableResultResource] = []
    for variable2, mask2 in zip(category_names, category_masks):
      anti_mask1 = ~mask1
      anti_mask2 = ~mask2
      TT = (mask1 & mask2).sum()
      TF = (mask1 & anti_mask2).sum()
      FT = (anti_mask1 & mask2).sum()
      FF = (anti_mask1 & anti_mask2).sum()
      contingency_table = pd.DataFrame([
        [TT, TF],
        [FT, FF],
      ], index=[variable1, f"NOT {variable1}"], columns=[variable2, f"NOT {variable2}"])

      warnings = []
      has_zero_values = (contingency_table == 0).sum().sum() > 0
      has_less_than_five = (contingency_table < 5).sum().sum() > 0
      if has_zero_values:
        contingency_table += 0.5
      if has_less_than_five:
        warnings.append(f"There's a cell that contains a frequency less than 5 ({contingency_table.min().min()}). Results may not be accurate.")

      OR = (TT * FF) / (TF * FT)
      Q = (OR - 1) / (OR + 1)

      chisq_result = scipy.stats.chi2_contingency(contingency_table)

      results_row.append(BinaryStatisticTestOnContingencyTableResultResource(
        warnings=warnings,
        effect_size=EffectSizeResult(
          type="yule_q",
          value=Q,
        ),
        significance=SignificanceResult(
          type=StatisticTestMethodEnum.ChiSquared,
          statistic=chisq_result.statistic, # type: ignore
          p_value=chisq_result.pvalue # type: ignore
        ),
        TT=TT,
        TF=TF,
        FT=FT,
        FF=FF,
        discriminator1=str(variable1),
        discriminator2=str(variable2),
      ))
    results.append(results_row)

  return BinaryStatisticTestOnContingencyTableResultMainResource(
    results=results,
    rows=subdataset_names,
    columns=category_names,
    column=column,
  )


__all__ = [
  "binary_statistic_test_on_distribution",
  "binary_statistic_test_on_contingency_table"
]