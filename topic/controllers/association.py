from typing import Sequence, cast

import pandas as pd
from common.ipc.client import IntraProcessCommunicator
from common.ipc.requests import IPCRequest, IPCRequestData, TopicSimilarityVisualizationMethod
import plotly.express

from common.ipc.responses import AssociationData, IPCResponseData
from common.ipc.task import IPCTask, TaskStepTracker, ipc_task_handler
from common.models.api import ApiError
from common.utils.iterable import array_find
from topic.controllers.utils import assert_column_exists

import wordsmith.stats
from wordsmith.data.config import Config
from wordsmith.data.paths import ProjectPaths
from wordsmith.data.schema import SchemaColumnType

def categorical_association_plot(a: pd.Series, b: pd.Series):
  indexed_residual_table = wordsmith.stats.indexed_residual_table(a, b)
  crosstab = pd.crosstab(a, b)
  normalized_crosstab = wordsmith.stats.normalize_frequency(crosstab, axis=0)

  association_heatmap = plotly.express.imshow(indexed_residual_table, aspect="auto")
  crosstab_heatmap = plotly.express.imshow(crosstab, aspect="auto")
  shared_params = dict(
    xaxis=dict(
      title=b.name,
      tickangle=30,
    ),
    yaxis=dict(
      title=a.name,
    ),
  )
  crosstab_heatmap.update_layout(
    shared_params,
    title=f"{str(a.name).capitalize()} x {str(b.name).capitalize()} (Frequency)",
  )
  crosstab_heatmap.update_traces(
    customdata=normalized_crosstab,
    hovertemplate="<br>".join([
      str(b.name).capitalize() + ": %{x}",
      str(a.name).capitalize() + ": %{y}",
      "Frequency: %{z}",
      "Percentage: %{customdata[x][y]}%",
    ])
  )
  association_heatmap.update_layout(
    shared_params,
    title=f"{str(a.name).capitalize()} x {str(b.name).capitalize()} (Association)",
    customdata=crosstab,
    hovertemplate="<br>".join([
      str(b.name).capitalize() + ": %{x}",
      str(a.name).capitalize() + ": %{y}",
      "Strength: %{z}",
      "Frequency: %{customdata[x][y]}%",
    ])
  )
  return IPCResponseData.Association(
    data=AssociationData.Categorical(
      crosstab_heatmap=cast(str, crosstab_heatmap.to_json()),
      association_heatmap=cast(str, association_heatmap.to_json()),
      biplot='',

      association_csv=indexed_residual_table.to_csv(),
      crosstab_csv=crosstab.to_csv(),

      topics=tuple(map(str, crosstab.columns)),
      outcomes=tuple(map(str, crosstab.index)),
    )
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
    title=f"{str(a.name).capitalize()} x {str(b.name).capitalize()} Association"
  )

  topics = tuple(map(str, df[a.name].cat.categories))
  statistics = {}
  for topic in topics:
    view: pd.Series = df[a.name] == topic
    rawnumbers = df.loc[view, str(b.name)]
    statistics[topic] = rawnumbers.describe()

  return IPCResponseData.Association(
    data=AssociationData.Continuous(
      statistics_csv=pd.DataFrame(statistics).to_csv(),
      plot=cast(str, violinplot.to_json()),
      topics=topics,
    )
  )

def temporal_association_plot(a: pd.Series, b: pd.Series, config: Config):
  model = config.paths.load_bertopic(str(a.name))
  col1_data = cast(list[str], a)
  col2_data = cast(list[str], b.astype(str))
  topics_over_time = model.topics_over_time(col1_data, col2_data)
  topic_plot = model.visualize_topics_over_time(topics_over_time, title=f"{str(a.name).capitalize()} Topics Over Time {str(b.name).capitalize()}")
  return IPCResponseData.Association(
    data=AssociationData.Temporal(
      bins=tuple(map(str, topics_over_time["Bins"])),
      topics=pd.Categorical(col1_data).categories,
      plot=cast(str, topic_plot.to_json()),
    )
  )

@ipc_task_handler
def association_plot(task: IPCTask):
  steps = TaskStepTracker(
    max_steps = 3,
  )

  task.progress(0, "Loading workspace table.")

  message = cast(IPCRequestData.AssociationPlot, task.request)
  config = Config.from_project(message.project_id)
  df = config.paths.load_workspace()

  task.progress(steps.advance(), f"Checking to make sure that {message.col1} and {message.col2} actually exist in the table.")

  col1_schema = config.dfschema.assert_exists(message.col1)
  col2_schema = config.dfschema.assert_exists(message.col2)

  if message.col1 == message.col2:
    raise ApiError(f"Both columns are the same ({message.col2}). Please select a different column to compare.", 422)

  if col1_schema.type != SchemaColumnType.Textual:
    raise ApiError("Only textual columns can be used as the left column. Please select a different column for the left side of the comparison.", 422)
  
  if col2_schema.type == SchemaColumnType.Unique:
    raise ApiError(f"Columns of type {SchemaColumnType.Unique.name} ({message.col2}) cannot be compared with any other columns due to their unique nature. Consider changing the type of {message.col2} to {SchemaColumnType.Categorical.name} if you need to analyze that column.", 422)
  
  col1_data = assert_column_exists(df, col1_schema.topic_column) \
    if col1_schema.type == SchemaColumnType.Textual \
    else assert_column_exists(df, message.col1)
  
  col2_data = assert_column_exists(df, col2_schema.topic_column) \
    if col2_schema.type == SchemaColumnType.Textual \
    else assert_column_exists(df, message.col1)
  
  task.progress(steps.advance(), f"Finding association between {message.col1} and {message.col2}")

  if col2_schema.type == SchemaColumnType.Categorical or col2_schema.type == SchemaColumnType.Textual:
    plot = categorical_association_plot(col1_data, col2_data)
  elif col2_schema.type == SchemaColumnType.Continuous:
    plot = continuous_association_plot(col1_data, col2_data)
  elif col2_schema.type == SchemaColumnType.Temporal:
    plot = temporal_association_plot(col1_data, col2_data, config)
  else:
    raise ApiError(f"The type of {message.col2} as registered in the configuration is invalid. Perhaps the configuration file was corrupted and had been modified in an incorrect manner. Please recreate this project or manually fix the fields in {config.paths.full_path(ProjectPaths.Config)}", 400)
  
  task.success(plot, None)

__all__ = [
  "association_plot",
]