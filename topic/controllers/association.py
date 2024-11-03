from typing import Sequence, cast

import numpy as np
import pandas as pd
from common.ipc.requests import IPCRequestData
import plotly.express

from common.ipc.responses import AssociationData, IPCResponseData
from common.ipc.task import IPCTask, TaskStepTracker
from common.models.api import ApiError
from common.utils.string import truncate_strings
from topic.controllers.utils import assert_column_exists

import wordsmith.stats
from wordsmith.data.config import Config
from wordsmith.data.paths import ProjectPaths
from wordsmith.data.schema import SchemaColumnTypeEnum, TemporalSchemaColumn, TextualSchemaColumn
import wordsmith.visual

def categorical_association_plot(a: pd.Series, b: pd.Series):
  residual_table, observed, expected = wordsmith.stats.residual_table(a, b)
  pearson_residuals = residual_table / np.sqrt(expected)
  association_table = pd.DataFrame(pearson_residuals, index=residual_table.index, columns=residual_table.columns)

  crosstab = pd.crosstab(a, b)
  normalized_crosstab = wordsmith.stats.normalize_frequency(crosstab, axis=None)
  heatmap_customdata = (normalized_crosstab * 100).map(lambda x: f"{x:.4f}")

  crosstab_clustergram, crosstab_sorter = wordsmith.visual.chart.clustergram(crosstab)
  association_clustergram, association_sorter = wordsmith.visual.chart.clustergram(association_table)
  shared_params = dict(
    xaxis=dict(
      title=b.name,
      ticktext=tuple(truncate_strings(cast(Sequence[str], crosstab.columns))),
    ),
    yaxis=dict(
      title=a.name,
      ticktext=tuple(truncate_strings(cast(Sequence[str], crosstab.index))),
    ),
  )
  crosstab_clustergram.update_layout(
    shared_params,
    title=f"{a.name} x {b.name} (Frequency)",
  )
  association_clustergram.update_layout(
    shared_params,
    title=f"{a.name} x {b.name} (Association)",
  )

  crosstab_clustergram.update_traces(
    customdata=heatmap_customdata.loc[crosstab_sorter],
    hovertemplate="<br>".join([
      str(b.name) + ": %{x}",
      str(a.name) + ": %{y}",
      "Frequency: %{z} (%{customdata}%)",
    ])
  )

  association_clustergram.update_traces(
    customdata=residual_table.loc[association_sorter],
    hovertemplate="<br>".join([
      str(b.name) + ": %{x}",
      str(a.name) + ": %{y}",
      "Strength: %{z}",
      "Residual: %{customdata}",
    ])
  )
  return AssociationData.Categorical(
    crosstab_heatmap=cast(str, crosstab_clustergram.to_json()),
    residual_heatmap=cast(str, association_clustergram.to_json()),
    biplot='',

    association_csv=association_table.to_csv(),
    crosstab_csv=crosstab.to_csv(),

    topics=tuple(map(str, crosstab.index)),
    outcomes=tuple(map(str, crosstab.columns)),
  )

def continuous_association_plot(a: pd.Series, b: pd.Series):
  df = pd.DataFrame({
    a.name: pd.Categorical(a),
    b.name: b,
  })
  violinplot = plotly.express.violin(
    df,
    x=a.name,
    y=b.name,
    box=True,
    title=f"{a.name} x {b.name} Association"
  )
  violinplot.update_layout(dict(
    xaxis=dict(
      ticktext=tuple(truncate_strings(cast(Sequence[str], a)))
    )
  ))

  topics = tuple(map(str, df[a.name].cat.categories))
  statistics = {}
  for topic in topics:
    view: pd.Series = df[a.name] == topic
    rawnumbers = df.loc[view, str(b.name)]
    statistics[topic] = rawnumbers.describe()

  return AssociationData.Continuous(
    statistics_csv=pd.DataFrame(statistics).to_csv(),
    violin_plot=cast(str, violinplot.to_json()),
    topics=topics
  )
    

def temporal_association_plot(a: pd.Series, b: pd.Series, config: Config, column_a: TextualSchemaColumn, column_b: TemporalSchemaColumn, mask: pd.Series):
  model = config.paths.load_bertopic(str(a.name))
  df = config.paths.load_workspace()

  documents = cast(list[str], df.loc[~mask, column_a.preprocess_column])
  topics: list[int] = list(df.loc[~mask, column_a.topic_index_column])

  col2_data = cast(list[str], b.astype(str))

  params = dict()
  if column_b.bins is not None:
    params["nr_bins"] = column_b.bins
  if column_b.datetime_format is not None:
    params["datetime_format"] = column_b.datetime_format

  topics_over_time = model.topics_over_time(documents, col2_data, topics, **params)
  topic_plot = model.visualize_topics_over_time(topics_over_time, topics=topics, title=f"{a.name} Topics Over Time {b.name}")

  return AssociationData.Temporal(
    topics=tuple(pd.Categorical(a).categories),
    line_plot=cast(str, topic_plot.to_json()),
  )

def association_plot(task: IPCTask):
  steps = TaskStepTracker(
    max_steps = 3,
  )

  task.progress(0, "Loading workspace table.")

  message = cast(IPCRequestData.Association, task.request)
  config = Config.from_project(message.project_id)
  df = config.paths.load_workspace()

  task.progress(steps.advance(), f"Checking to make sure that {message.column1} and {message.column2} actually exist in the table.")

  col1_schema = config.data_schema.assert_exists(message.column1)
  col2_schema = config.data_schema.assert_exists(message.column2)

  if message.column1 == message.column2:
    raise ApiError(f"Both columns are the same ({message.column2}). Please select a different column to compare.", 422)

  if col1_schema.type != SchemaColumnTypeEnum.Textual:
    raise ApiError("Only textual columns can be used as the left column. Please select a different column for the left side of the comparison.", 422)
  
  if col2_schema.type == SchemaColumnTypeEnum.Unique:
    raise ApiError(f"Columns of type {SchemaColumnTypeEnum.Unique.name} ({message.column2}) cannot be compared with any other columns due to their unique nature. Consider changing the type of {message.column2} to {SchemaColumnTypeEnum.Categorical.name} if you need to analyze that column.", 422)
  
  col1_data = assert_column_exists(df, col1_schema.topic_column)
  col1_data.name = col1_schema.name

  col2_data = assert_column_exists(df, col2_schema.topic_column) \
    if col2_schema.type == SchemaColumnTypeEnum.Textual \
    else assert_column_exists(df, message.column2)
  col2_data.name = col2_schema.name
  
  mask1 = col1_data == ''
  mask2 = col2_data.isna() | (col2_data == '')
  mask = mask1 | mask2
  col1_data = col1_data[~mask]
  col2_data = col2_data[~mask]

  excluded = np.count_nonzero(mask)
  excluded_left = np.count_nonzero(mask1)
  excluded_right = np.count_nonzero(mask2)
  
  task.progress(steps.advance(), f"Plotting association between {message.column1} and {message.column2}")

  if col2_schema.type == SchemaColumnTypeEnum.Categorical or col2_schema.type == SchemaColumnTypeEnum.Textual:
    plot = categorical_association_plot(col1_data, col2_data)
  elif col2_schema.type == SchemaColumnTypeEnum.Continuous:
    plot = continuous_association_plot(col1_data, col2_data)
  elif col2_schema.type == SchemaColumnTypeEnum.Temporal:
    plot = temporal_association_plot(col1_data, col2_data, config, col1_schema, col2_schema, mask)
  else:
    raise ApiError(f"The type of {message.column2} as registered in the configuration is invalid. Perhaps the configuration file was corrupted and had been modified in an incorrect manner. Please recreate this project or manually fix the fields in {config.paths.full_path(ProjectPaths.Config)}", 400)
  
  task.success(IPCResponseData.Association(
    data=plot,
    column1=str(col1_schema.name),
    column2=str(col2_schema.name),
    excluded=excluded,
    excluded_left=excluded_left,
    excluded_right=excluded_right,
    total=len(df.index),
  ), None)

__all__ = [
  "association_plot",
]