import pandas as pd
from modules.comparison.engine import TableComparisonEngine, StatisticTestResult
from modules.config.schema.base import ANALYZABLE_SCHEMA_COLUMN_TYPES
from modules.logger.provisioner import ProvisionedLogger
from modules.project.cache import ProjectCache
from routes.comparison.model import BinaryStatisticTestSchema, PairwiseStatisticTestResource, PairwiseStatisticTestSchema
from routes.table.controller.preprocess import TablePreprocessModule

def pairwise_statistic_test(cache: ProjectCache, input: PairwiseStatisticTestSchema):
  preprocess = TablePreprocessModule(cache)
  column = preprocess.assert_column(input.column, supported_types=ANALYZABLE_SCHEMA_COLUMN_TYPES)
  workspace = cache.workspaces.load()

  results: list[list[StatisticTestResult]] = []
  with ProvisionedLogger().disable(["TableEngine", "TableComparisonEngine"]):
    for group1_idx, group1 in enumerate(input.groups):
      result_row: list[StatisticTestResult] = []
      for group2_idx, group2 in enumerate(input.groups):
        if group2_idx > group1_idx:
          break
        engine = TableComparisonEngine(config=cache.config, groups=[group1, group2])
        comparison_result = engine.compare(
          df=workspace,
          column_name=input.column,
          statistic_test_preference=input.statistic_test_preference,
          effect_size_preference=input.effect_size_preference,
        )
        result_row.append(comparison_result)
      results.append(result_row)
  return PairwiseStatisticTestResource(
    column=column,
    results=results,
    groups=list(map(lambda group: group.name, input.groups)),
  )



def binary_statistic_test_on_distribution(cache: ProjectCache, input: BinaryStatisticTestSchema):
  preprocess = TablePreprocessModule(cache)
  column = preprocess.assert_column(input.column, supported_types=ANALYZABLE_SCHEMA_COLUMN_TYPES)
  workspace = cache.workspaces.load()
  total_count = len(workspace)

  TableComparisonEngine(config=cache.config, groups=input.groups)

  discriminator = preprocess.extract(df, partial1.column, transform_topics=False)
  binary_variables = discriminator.unique()
  binary_variable_labels = preprocess.label_binary_variable(binary_variables, partial1.column)
  if len(binary_variables) == 0:
    raise ApiError(f"{partial1.column.name} does not contain any values that can be used to discriminate {partial2.column.name}.", HTTPStatus.BAD_REQUEST)

  results: list[BinaryStatisticTestOnDistributionResource] = []
  discriminated_groups: list[pd.Series] = []
  with ProvisionedLogger().disable(["TableEngine", "TableComparisonEngine"]):
    for variable, label in zip(binary_variables, binary_variable_labels):
      filter = EqualToTableFilter(target=input.column1, value=variable)
      anti_filter = NotTableFilter(operand=filter)
      treatment_group = NamedTableFilter(name=label, filter=filter)
      control_group = NamedTableFilter(name=label, filter=anti_filter)
      table_engine = TableEngine(config=cache.config)

      discriminated_df = table_engine.process_workspace(filter, None)
      discriminated_group = discriminated_df[input.column2]
      discriminated_group.name = label
      discriminated_groups.append(discriminated_group)
      comparison_engine = TableComparisonEngine(
        config=cache.config,
        engine=table_engine,
        groups=[treatment_group, control_group],
      )
      try:
        result = comparison_engine.compare(
          df,
          column_name=input.column2,
          statistic_test_preference=input.statistic_test_preference,
          effect_size_preference=input.effect_size_preference,
        )
      except EmptyComparisonGroupException as e:
        continue

      yes_count = result.groups[0].valid_count
      no_count = result.groups[1].valid_count
      results.append(BinaryStatisticTestOnDistributionResource(
        warnings=result.warnings,
        effect_size=result.effect_size,
        significance=result.significance,
        yes_count=yes_count,
        no_count=no_count,
        invalid_count=total_count - yes_count - no_count,
        discriminator=label,
      ))

  if len(results) == 0:
    raise ApiError(f"There are no valid subdatasets that can be made using {partial1.column.name}.", HTTPStatus.UNPROCESSABLE_ENTITY)
  