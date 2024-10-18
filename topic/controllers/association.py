from typing import Sequence, cast

import pandas as pd
from common.ipc.requests import IPCRequestData, TopicSimilarityVisualizationMethod
import plotly.express

from common.ipc.responses import IPCResponseData
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
    title=f"{a.name} x {b.name} (Frequency)",
  )
  crosstab_heatmap.update_traces(
    customdata=normalized_crosstab,
    hovertemplate="<br>".join([
      str(b.name) + ": %{x}",
      str(a.name) + ": %{y}",
      "Frequency: %{z}",
      "Percentage: %{customdata[x][y]}%",
    ])
  )
  association_heatmap.update_layout(
    shared_params,
    title=f"{a.name} x {b.name} (Association)",
    customdata=crosstab,
    hovertemplate="<br>".join([
      str(b.name) + ": %{x}",
      str(a.name) + ": %{y}",
      "Strength: %{z}",
      "Frequency: %{customdata[x][y]}%",
    ])
  )
  return IPCResponseData.CategoricalAssociationPlot(
    crosstab_heatmap=cast(str, crosstab_heatmap.to_json()),
    association_heatmap=cast(str, association_heatmap.to_json()),
    biplot='',

    association=indexed_residual_table.to_numpy(),
    crosstab=crosstab.to_numpy(),

    xaxis=tuple(map(str, crosstab.columns)),
    yaxis=tuple(map(str, crosstab.index)),
  )


def association_plot(message: IPCRequestData.AssociationPlot):
  config = Config.from_project(message.project_id)
  df = config.paths.load_workspace()
  
  col1_schema = config.dfschema.assert_exists(message.col1)
  col2_schema = config.dfschema.assert_exists(message.col2)


  categorical_comparison = (SchemaColumnType.Textual, SchemaColumnType.Categorical)
  if col1_schema.type not in categorical_comparison:
    raise ApiError("Only textual/categorical columns can be used as the left column. Please select a different column for the left side of the comparison.", 422)
  
  if col2_schema.type == SchemaColumnType.Unique:
    raise ApiError(f"Columns of type {SchemaColumnType.Unique.name} ({message.col2}) cannot be compared with any other columns due to their unique nature. Consider changing the type of {message.col2} to {SchemaColumnType.Categorical.name} if you need to analyze that column.", 422)
  
  if col1_schema.type == SchemaColumnType.Textual:
    col1_data = assert_column_exists(df, col1_schema.topic_column)
  else:
    col1_data = assert_column_exists(df, message.col1)

  if col2_schema.type == SchemaColumnType.Textual:
    col2_data = assert_column_exists(df, col2_schema.topic_column)
  else:
    col2_data = assert_column_exists(df, message.col1)

  if col2_schema.type in categorical_comparison:
    return categorical_association_plot(col1_data, col2_data)

  if col2_schema.type == SchemaColumnType.Continuous:
    return
  
  if col2_schema.type == SchemaColumnType.Temporal:
    model = config.paths.load_bertopic(message.col2)
    col1_data = cast(list[str], col1_data)
    col2_data = cast(list[str], col2_data.astype(str))
    topics_over_time = model.topics_over_time(col1_data, col2_data)
    return model.visualize_topics_over_time(topics_over_time, title=f"{message.col1.capitalize()} Topics Over Time {message.col2.capitalize()}")
  
  raise ApiError(f"The type of {message.col2} as registered in the configuration is invalid. Perhaps the configuration file was corrupted and had been modified in an incorrect manner. Please recreate this project or manually fix the fields in {config.paths.full_path(ProjectPaths.Config)}", 400)

